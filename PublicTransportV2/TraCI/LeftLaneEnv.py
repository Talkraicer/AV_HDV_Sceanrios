from multiprocessing import Pool

import gymnasium as gym
import numpy as np
import sys

from tqdm import tqdm

from utils import handle_step, log_features, exp_name
from simulation_run import init_simulation, NUM_PROCESSES
import traci
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

ACT_RATE = 100
TrainTimeSteps = 100000
# DECISION VARיIABLE
MIN_NUM_PASS = 1

# List of relevant studied features
OBSERVATIONS = ["num_total_vehs", "num_hdv_in_end_PTL", "num_vehs_in_PTL",
                "mean_speed_in_end_PTL", "mean_speed"
    , "time_step", "av_rate"]  # last row is for dummy observation
NUM_DUMMY_OBS = 3


def action_wrapper(state, gym_policy_name, policy_name):
    # run the step
    av_rate = state[-1]
    t_start = state[-2]

    for i in range(ACT_RATE):
        handle_step(t_start, policy_name, "av"+str(av_rate))
        t_start += 1
        traci.simulationStep(t_start)

    # get the new state
    state[-2] += ACT_RATE
    t_end = state[-2]
    new_features = log_features(gym_policy_name + exp_name + "_av" + str(av_rate) + ".xml", t_end, ACT_RATE)
    if new_features:
        for i in range(len(OBSERVATIONS) - NUM_DUMMY_OBS):
            state[i] = new_features[OBSERVATIONS[i]]
        reward = 1 / new_features["mean_pass_delay_timestamp"]
    else:
        reward = 0

    done = traci.simulation.getMinExpectedNumber() <= 0

    return reward, done


ACTIONS = [lambda state, policy_name: action_wrapper(state, policy_name, f"StaticNumPassFL_{i}") for i in range(1, 6)]

ACTION_SPACE = gym.spaces.Discrete(len(ACTIONS))


def make_observation_space():
    return gym.spaces.Box(low=0, high=100000, shape=(len(OBSERVATIONS),))


class LeftLaneENV(gym.Env):
    def __init__(self, policy_name, sumoCfg):
        self.observation_space = make_observation_space()
        self.action_space = ACTION_SPACE
        self.state = None
        self.av_rate = None
        self.policy_name = policy_name
        self.sumoCfg = sumoCfg
        self.log = ""
        self.act_rate = ACT_RATE

    def observation(self):
        return np.array([self.state[i] for i in range(len(OBSERVATIONS))])

    def reset(self, seed=None, options=None, ):
        # check if a traci instance is already running

        _, _, self.av_rate = init_simulation((self.policy_name, self.sumoCfg))
        self.av_rate = self.av_rate[2:]
        self.state = [0] * len(OBSERVATIONS)
        self.state[-1] = float(self.av_rate)
        return self.observation(), {}

    def step(self, action):
        reward, done = ACTIONS[action](self.state, self.policy_name)
        self.log += f"Time: {self.state[-2] - self.act_rate} Action: {action + 1}, Reward: {reward}\n"
        if done:
            traci.close()
        return self.observation(), reward, done, False, {}

    def render(self, mode=None):
        print(self.log)
        self.log = ''


def train_agent(sumoCfg):
    # Register the environment with Gym
    gym.envs.registration.register(
        id='LeftLaneENV-v0',
        entry_point=LeftLaneENV,
        max_episode_steps=1000,
    )
    # env = make_vec_env('LeftLaneENV-v0', n_envs=1,
    #                    env_kwargs={"policy_name": "DQNAgent", "sumoCfg": sumoCfg})
    env = gym.make('LeftLaneENV-v0', policy_name="DQNAgent", sumoCfg=sumoCfg)
    model = DQN("MlpPolicy", env, verbose=1)

    model.learn(total_timesteps=TrainTimeSteps)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

    agent_name = ".".join(sumoCfg.split("/")[-1].split(".")[:-1]) + "_DQN"
    model.save(agent_name)

    env.close()


if __name__ == '__main__':
    sumoCfgs = [f"../cfg_files_LeftCompDaily/LeftCompDaily_av{r}.sumocfg" for r in [0.2, 0.4, 0.6, 0.8]]
    with Pool(min(NUM_PROCESSES,len(sumoCfgs))) as pool:
        tqdm(pool.map(train_agent, sumoCfgs), total=len(sumoCfgs))