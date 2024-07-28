from multiprocessing import Pool
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import wandb
from utils import handle_step, log_features, exp_name, init_wandb_logger
from simulation_run import init_simulation, NUM_PROCESSES
import traci
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

ACT_RATE = 300
TrainTimeSteps = 10000
NUM_EDGES = 8

# DECISION VARIABLE
MIN_NUM_PASS = 1

# List of relevant studied features
OBSERVATIONS = []

def set_observations(obs_type):
    global OBSERVATIONS
    if obs_type == "LOG_FEATURES":
        OBSERVATIONS = ["num_total_vehs", "num_hdv_in_end_PTL", "num_vehs_in_PTL",
                "mean_speed_in_end_PTL", "mean_speed"]

    if obs_type == "E_FEATURES":
        for i in range(NUM_EDGES):
            OBSERVATIONS.append(f"num_vehs_edge_{i}_no_PTL")
            OBSERVATIONS.append(f"mean_speed_edge_{i}_no_PTL")
            OBSERVATIONS.append(f"num_vehs_edge_{i}_in_PTL")
            OBSERVATIONS.append(f"mean_speed_edge_{i}_in_PTL")


    if obs_type == "LOG_FEATURES":
        return gym.spaces.Box(low=0, high=100000, shape=(len(OBSERVATIONS),))


def action_wrapper(env, policy_name):
    # run the step

    for i in range(ACT_RATE):
        handle_step(env.timestep, policy_name, "av"+str(env.av_rate),log_rate=0)
        env.timestep += 1
        traci.simulationStep(env.timestep)

    # get the new state
    if env.obs_type == "LOG_FEATURES":
        new_features = log_features(env.policy_name + exp_name + "_av" + str(env.av_rate) + ".xml", env.timestep, ACT_RATE)
        if new_features:
            for i in range(len(OBSERVATIONS)):
                env.state[i] = new_features[OBSERVATIONS[i]]
            reward = 1 / new_features["mean_pass_delay_timestamp"]
        else:
            reward = 0

    done = traci.simulation.getMinExpectedNumber() <= 0

    return reward, done, new_features


ACTIONS = [lambda env: action_wrapper(env,f"StaticNumPassFL_{i}") for i in range(1, 7)]

ACTION_SPACE = gym.spaces.Discrete(len(ACTIONS))



class LeftLaneENV(gym.Env):
    def __init__(self, policy_name, sumoCfg, log_wandb=True, features_type="LOG_FEATURES"):
        self.observation_space = set_observations(features_type)
        self.action_space = ACTION_SPACE
        self.state = None
        self.timestep = 0
        self.av_rate = ".".join(sumoCfg.split("/")[-1].split(".")[:-1]).split("_")[-1][2:]
        self.policy_name = policy_name
        self.sumoCfg = sumoCfg
        self.log_wandb = log_wandb
        if log_wandb:
            init_wandb_logger(self.policy_name,"av"+str(self.av_rate),delete_older=True)
        self.log = ""
        self.act_rate = ACT_RATE

    def observation(self):
        return np.array([self.state[i] for i in range(len(OBSERVATIONS))])

    def reset(self, seed=None, options=None, ):
        # check if a traci instance is already running
        self.timestep = 0
        _, _, self.av_rate = init_simulation((self.policy_name, self.sumoCfg))
        self.av_rate = float(self.av_rate[2:])
        self.state = [0] * len(OBSERVATIONS)

        for i in range(20):
            ACTIONS[5](self)

        return self.observation(), {}

    def step(self, action):
        reward, done, log_msg = ACTIONS[action](self)
        self.log += f"Time: {self.timestep - self.act_rate} Action: {action + 1}, Reward: {reward}\n"
        if self.log_wandb:
            log_msg["MinNumPass"] = action + 1
            log_msg["Reward"] = reward
            wandb.log(log_msg)
        if done:
            traci.close()
        return self.observation(), reward, done, False, {}

    def render(self, mode=None):
        print(self.log)
        self.log = ''


def train_agent(sumoCfg,policy_name = "DQNAgentV2"):
    # Register the environment with Gym
    gym.envs.registration.register(
        id='LeftLaneENV-v0',
        entry_point=LeftLaneENV,
        max_episode_steps=1000,
    )

    env = gym.make('LeftLaneENV-v0', policy_name=policy_name, sumoCfg=sumoCfg)
    model = DQN("MlpPolicy", env, verbose=1)

    model.learn(total_timesteps=TrainTimeSteps)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

    agent_name = ".".join(sumoCfg.split("/")[-1].split(".")[:-1]) + policy_name
    model.save(agent_name)

    env.close()


if __name__ == '__main__':
    sumoCfgs = [f"../cfg_files_LeftCompDaily/LeftCompDaily_av{r}.sumocfg" for r in [0.2, 0.4, 0.6, 0.8]]
    with Pool(min(NUM_PROCESSES,len(sumoCfgs))) as pool:
        tqdm(pool.map(train_agent, sumoCfgs), total=len(sumoCfgs))