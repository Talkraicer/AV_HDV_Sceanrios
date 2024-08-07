import os
from multiprocessing import Pool

import gymnasium as gym
from stable_baselines3 import PPO, DQN, A2C
import wandb
from LeftLaneEnv import LeftLaneENV
from utils import init_wandb_logger, log_features
def infer(agent_path):
    agent_path_cln = agent_path.split("/")[1]
    exp_name = agent_path_cln.split("_")[0]
    agent_type_str = agent_path_cln.split("_")[2]
    if agent_type_str == "PPO":
        agent = PPO.load(agent_path)
    elif agent_type_str == "DQN":
        agent = DQN.load(agent_path)
    elif agent_type_str == "A2C":
        agent = A2C.load(agent_path)
    else:
        raise ValueError("Invalid agent path")
    av_rate_str = agent_path_cln.split("_")[1]
    feat_type = "_".join(agent_path_cln.split("_")[3:6])
    act_rate = agent_path_cln.split("_")[-3]
    act_type = agent_path_cln.split("_")[-2]
    sumo_cfg = f"../cfg_files_{exp_name}/{exp_name}_{av_rate_str}.sumocfg"

    gym.envs.registration.register(
        id='LeftLaneENV-v0',
        entry_point=LeftLaneENV,
    )
    env = gym.make('LeftLaneENV-v0', policy_name =agent_path_cln, sumoCfg = sumo_cfg, action_space_tag = act_type,
                   log_wandb = False, features_type = feat_type, act_rate = int(act_rate))
    obs, _ = env.reset()
    done = False
    init_wandb_logger(policy_name=agent_path_cln, av_rate=av_rate_str, delete_older=True)
    left_env = env.env.env
    while not done:
        log_msg = log_features(left_env.policy_name + exp_name + "_av" + str(left_env.av_rate) + ".xml", left_env.timestep, left_env.act_rate)
        if log_msg:
            log_msg["MinNumPass"] = left_env.last_action
            wandb.log(log_msg)
        action, _ = agent.predict(obs)
        obs, reward, done, _ , _ = env.step(action)

if __name__ == '__main__':
    # paths = [f"agents/{path}" for path in os.listdir("agents") if path != "logged_agents"]
    paths = ["agents/LeftCompDaily_av0.4_DQN_MS_EPTL_ACT_RATE_100_alter_3769.0"]
    with Pool() as p:
        p.map(infer, paths)