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

it_len = 10000
num_it = 100
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
            if 1 <= i < NUM_EDGES - 1:
                OBSERVATIONS.append(f"num_vehs_edge_{i}_in_PTL")
                OBSERVATIONS.append(f"mean_speed_edge_{i}_in_PTL")

    if obs_type == "E_FEATURES":
        return gym.spaces.Box(low=0, high=200, shape=(len(OBSERVATIONS),))

    if obs_type == "LOG_FEATURES":
        return gym.spaces.Box(low=0, high=100000, shape=(len(OBSERVATIONS),))


def action_wrapper(env, policy_name):
    # run the step

    for i in range(env.act_rate):
        handle_step(env.timestep, policy_name, "av" + str(env.av_rate), log_rate=0)
        env.timestep += 1
        traci.simulationStep(env.timestep)

    # get the new state
    new_features = log_features(env.policy_name + exp_name + "_av" + str(env.av_rate) + ".xml", env.timestep,
                                env.act_rate)
    if new_features:
        if env.features_type == "LOG_FEATURES":
            for i in range(len(OBSERVATIONS)):
                env.state[i] = new_features[OBSERVATIONS[i]]
        reward = 1 / new_features["mean_pass_delay"]
    else:
        reward = 0

    if env.features_type == "E_FEATURES":
        for e_idx in range(NUM_EDGES):
            edge = "E" + str(e_idx)
            if 1 <= e_idx < NUM_EDGES - 1:
                num_lanes = traci.edge.getLaneNumber(edge)
                PTL_idx = num_lanes - 1
                env.state[OBSERVATIONS.index(f"num_vehs_edge_{e_idx}_in_PTL")] = \
                    traci.lane.getLastStepVehicleNumber(f"{edge}_{PTL_idx}")
                env.state[OBSERVATIONS.index(f"mean_speed_edge_{e_idx}_in_PTL")] = \
                    traci.lane.getLastStepMeanSpeed(f"{edge}_{PTL_idx}")

                num_vehs_no_PTL = [traci.lane.getLastStepVehicleNumber(f"{edge}_{i}") for i in range(PTL_idx)]
                mean_speed_no_PTL = [traci.lane.getLastStepMeanSpeed(f"{edge}_{i}") for i in range(PTL_idx)]
                num_vehs_no_PTL = sum(num_vehs_no_PTL)
                mean_speed_no_PTL = sum(mean_speed_no_PTL) / (num_lanes - 1)
                env.state[OBSERVATIONS.index(f"num_vehs_edge_{e_idx}_no_PTL")] = num_vehs_no_PTL
                env.state[OBSERVATIONS.index(f"mean_speed_edge_{e_idx}_no_PTL")] = mean_speed_no_PTL

            else:
                env.state[OBSERVATIONS.index(f"num_vehs_edge_{e_idx}_no_PTL")] = \
                    traci.edge.getLastStepVehicleNumber(edge)
                env.state[OBSERVATIONS.index(f"mean_speed_edge_{e_idx}_no_PTL")] = \
                    traci.edge.getLastStepMeanSpeed(edge)

    done = traci.simulation.getMinExpectedNumber() <= 0

    return reward, done, new_features


ACTIONS = [lambda env: action_wrapper(env, f"StaticNumPassFL_{i}") for i in range(1, 7)]

ACTION_SPACE = gym.spaces.Discrete(len(ACTIONS))


class LeftLaneENV(gym.Env):
    def __init__(self, policy_name, sumoCfg, log_wandb=True, features_type="LOG_FEATURES", act_rate = 1):
        self.observation_space = set_observations(features_type)
        self.features_type = features_type
        self.actions = ACTIONS
        self.action_space = ACTION_SPACE
        self.state = None
        self.timestep = 0
        self.av_rate = ".".join(sumoCfg.split("/")[-1].split(".")[:-1]).split("_")[-1][2:]
        self.policy_name = policy_name
        self.sumoCfg = sumoCfg
        self.log_wandb = log_wandb
        self.act_rate = act_rate
        if log_wandb:
            proj_tail = "_ACT_RATE_"+str(self.act_rate)
            init_wandb_logger(self.policy_name, "av" + str(self.av_rate)+proj_tail, delete_older=True)
        self.log = ""

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
        reward, done, log_msg = action_wrapper(self, "StaticNumPassFL_" + str(action + 1))
        self.log += f"Time: {self.timestep - self.act_rate} Action: {action + 1}, Reward: {reward}\n"
        if self.log_wandb and log_msg:
            log_msg["MinNumPass"] = action + 1
            log_msg["Reward"] = reward
            wandb.log(log_msg)
        if done:
            traci.close()
        return self.observation(), reward, done, False, {}

    def render(self, mode=None):
        print(self.log)
        self.log = ''


def train_agent(cfg, policy_name="DQN"):
    sumoCfg, feat_type, act_rate = cfg
    policy_name += "_"+feat_type+"_ACT_RATE_"+str(act_rate)
    # Register the environment with Gym
    gym.envs.registration.register(
        id='LeftLaneENV-v0',
        entry_point=LeftLaneENV,
    )

    env = gym.make('LeftLaneENV-v0', policy_name=policy_name, sumoCfg=sumoCfg, features_type=feat_type,
                   act_rate = act_rate)
    model = DQN("MlpPolicy", env, verbose=1)

    for i in range(num_it):
        model.learn(total_timesteps=it_len)

        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        print(f"it {i} Mean reward: {mean_reward} +/- {std_reward}")
        experiment_name = ".".join(sumoCfg.split("/")[-1].split(".")[:-1])
        agent_name = experiment_name +"_"+ policy_name
        model.save(agent_name+"_"+str(i))

    env.close()


if __name__ == '__main__':
    sumoCfgs = [f"../cfg_files_LeftCompDaily/LeftCompDaily_av{r}.sumocfg" for r in [0.2, 0.4, 0.6, 0.8]]
    feat_types = ["LOG_FEATURES", "E_FEATURES"]
    act_rates = [1,100,300]
    cfgs = [(sumoCfg, feat_type, act_rate) for sumoCfg in sumoCfgs for feat_type in feat_types for act_rate in act_rates]
    print("num cfgs", len(cfgs))
    with Pool(min(NUM_PROCESSES, len(cfgs))) as pool:
        tqdm(pool.map(train_agent, cfgs), total=len(sumoCfgs))
