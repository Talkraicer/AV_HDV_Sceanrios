import os
from multiprocessing import Pool
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import wandb
from utils import handle_step, log_features, exp_name, init_wandb_logger, output_file_to_df
from simulation_run import init_simulation, NUM_PROCESSES
import traci
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback

NUM_EDGES = 8
NUM_E_FEATURES = 6
MAX_SPEED = 25
num_it = 10000
# DECISION VARIABLE
MIN_NUM_PASS = 1

# List of relevant studied features
OBSERVATIONS = []


def set_observations(features_type, obs_type):
    global OBSERVATIONS
    high_limit = 0
    if features_type == "LOG_FEATURES":
        OBSERVATIONS = ["num_total_vehs", "num_hdv_in_end_PTL", "num_vehs_in_PTL",
                        "mean_speed_in_end_PTL", "mean_speed"]
        high_limit = 100000

    elif features_type in ["E_FEATURES", "E_FEATURES_TU"]:
        for i in range(NUM_EDGES):
            OBSERVATIONS.append(f"num_vehs_edge_{i}_no_PTL")
            OBSERVATIONS.append(f"mean_speed_edge_{i}_no_PTL")
            OBSERVATIONS.append(f"std_speed_edge_{i}_no_PTL")
            if 1 <= i < NUM_EDGES - 1:
                OBSERVATIONS.append(f"num_vehs_edge_{i}_in_PTL")
                OBSERVATIONS.append(f"mean_speed_edge_{i}_in_PTL")
                OBSERVATIONS.append(f"std_speed_edge_{i}_in_PTL")
        high_limit = 200

    elif features_type == "MS_EPTL":
        OBSERVATIONS = ["mean_speed_in_end_PTL"]
        high_limit = 1

    if obs_type == "vec":
        return gym.spaces.Box(low=0, high=high_limit, shape=(len(OBSERVATIONS),), dtype=np.float32)
    elif obs_type == "img":
        return gym.spaces.Box(low=0, high=255, shape=(1,NUM_E_FEATURES, NUM_EDGES), dtype=np.float32)


def calculate_reward(env):
    output_file = "results_reps/" +env.policy_name + exp_name + "_av" + str(env.av_rate) + ".xml"
    with open(output_file, "a+") as f:
        f.write("</tripinfos>")
    df = output_file_to_df(output_file)
    df_timestep = df[df["arrivalTime"] >= env.timestep - env.act_rate]
    # remove the <tripinfo> tag
    with open(output_file, "r") as f:
        lines = f.readlines()
    with open(output_file, "w") as f:
        f.writelines(lines[:-1])
    if df_timestep.empty:
        return 0
    total_delay = df_timestep.apply(lambda x: x["totalDelay"] * x["numPass"], axis=1).sum()
    return -total_delay / env.act_rate

def calc_E_features(env):
    for e_idx in range(NUM_EDGES):
        edge = "E" + str(e_idx)
        if 1 <= e_idx < NUM_EDGES - 1:
            num_lanes = traci.edge.getLaneNumber(edge)
            PTL_idx = num_lanes - 1
            PTL_speeds = [traci.vehicle.getSpeed(vehID) for vehID in
                        traci.lane.getLastStepVehicleIDs(f"{edge}_{PTL_idx}")]
            env.state[OBSERVATIONS.index(f"num_vehs_edge_{e_idx}_in_PTL")] = \
                traci.lane.getLastStepVehicleNumber(f"{edge}_{PTL_idx}")
            env.state[OBSERVATIONS.index(f"mean_speed_edge_{e_idx}_in_PTL")] = \
                traci.lane.getLastStepMeanSpeed(f"{edge}_{PTL_idx}")
            env.state[OBSERVATIONS.index(f"std_speed_edge_{e_idx}_in_PTL")] = \
                np.std(PTL_speeds)

        edge_vehIDs = traci.edge.getLastStepVehicleIDs(f"E{e_idx}")
        edge_speeds = [traci.vehicle.getSpeed(vehID) for vehID in edge_vehIDs]
        num_vehs_no_PTL = len(edge_vehIDs)
        mean_speed_no_PTL = np.mean(edge_speeds)
        std_speed_no_PTL = np.std(edge_speeds)
        env.state[OBSERVATIONS.index(f"num_vehs_edge_{e_idx}_no_PTL")] = num_vehs_no_PTL
        env.state[OBSERVATIONS.index(f"mean_speed_edge_{e_idx}_no_PTL")] = mean_speed_no_PTL
        env.state[OBSERVATIONS.index(f"std_speed_edge_{e_idx}_no_PTL")] = std_speed_no_PTL
def calc_E_features_TU():
    speeds = {}
    for e_idx in range(NUM_EDGES):
        edge = "E" + str(e_idx)
        if 1 <= e_idx < NUM_EDGES - 1:
            num_lanes = traci.edge.getLaneNumber(edge)
            PTL_idx = num_lanes - 1
            PTL_speeds = [traci.vehicle.getSpeed(vehID) for vehID in
                          traci.lane.getLastStepVehicleIDs(f"{edge}_{PTL_idx}")]
            speeds[f"edge_{e_idx}_in_PTL"] = PTL_speeds

        edge_vehIDs = traci.edge.getLastStepVehicleIDs(f"E{e_idx}")
        edge_speeds = [traci.vehicle.getSpeed(vehID) for vehID in edge_vehIDs]
        speeds[f"edge_{e_idx}_no_PTL"] = edge_speeds
    return speeds


def action_wrapper(env, policy_name):
    # run the step
    for i in range(env.act_rate):
        handle_step(env.timestep, policy_name, "av" + str(env.av_rate), log_rate=0)
        traci.simulationStep(env.timestep)
        env.timestep += 1
        if env.features_type == "E_FEATURES_TU":
            e_features = calc_E_features_TU()
            if i == 0:
                tot_speeds = e_features
            else:
                for key in tot_speeds.keys():
                    tot_speeds[key] += e_features[key]

    new_features = log_features(env.policy_name + exp_name + "_av" + str(env.av_rate) + ".xml", env.timestep,
                                env.act_rate)
    if env.features_type == "LOG_FEATURES":
        if new_features:
            # get the new state
            for i in range(len(OBSERVATIONS)):
                env.state[i] = new_features[OBSERVATIONS[i]]
    elif env.features_type == "MS_EPTL" and new_features:
        env.state[OBSERVATIONS.index("mean_speed_in_end_PTL")] = new_features["mean_speed_in_end_PTL"]/(MAX_SPEED*1.5)

    elif env.features_type == "E_FEATURES":
        calc_E_features(env)

    elif env.features_type == "E_FEATURES_TU":
        for obs in OBSERVATIONS:
            location = "_".join(obs.split("_")[2:])
            if obs.startswith("num"):
                env.state[OBSERVATIONS.index(obs)] = len(tot_speeds[location])/env.act_rate
            elif obs.startswith("mean"):
                env.state[OBSERVATIONS.index(obs)] = np.mean(tot_speeds[location]) if tot_speeds[location] else 0
            elif obs.startswith("std"):
                env.state[OBSERVATIONS.index(obs)] = np.std(tot_speeds[location]) if tot_speeds[location] else 0

    done = traci.simulation.getMinExpectedNumber() <= 0

    return calculate_reward(env), done, new_features



class LeftLaneENV(gym.Env):
    def __init__(self, policy_name, sumoCfg, action_space_tag="direct", log_wandb=True, features_type="LOG_FEATURES", act_rate=1):
        self.obs_type = "img" if policy_name.split("_")[1] == "CNN" else "vec"

        self.observation_space = set_observations(features_type, self.obs_type)
        self.features_type = features_type

        self.action_space_tag = action_space_tag
        self.actions = [lambda env: action_wrapper(env, f"StaticNumPassFL_{i}") for i in range(1, 7)]
        if action_space_tag == "direct":
            self.action_space = gym.spaces.Discrete(len(self.actions))
        elif action_space_tag == "alter":
            self.action_space = gym.spaces.Discrete(3)

        self.last_action = None
        self.state = None
        self.timestep = 0
        self.av_rate = ".".join(sumoCfg.split("/")[-1].split(".")[:-1]).split("_")[-1][2:]
        self.policy_name = policy_name
        self.sumoCfg = sumoCfg
        self.log_wandb = log_wandb
        self.act_rate = act_rate
        if log_wandb:
            proj_tail = "_ACT_RATE_" + str(self.act_rate)
            init_wandb_logger(self.policy_name, "av" + str(self.av_rate) + proj_tail+ "_" + action_space_tag, delete_older=True)
        self.log = ""

        self.model = None
        self.best_mean_pass_delay = 99999999
        self.agent_name = ".".join(sumoCfg.split("/")[-1].split(".")[:-1]) + "_" + self.policy_name

    def observation(self):
        if self.obs_type == "vec":
            return np.array([self.state[i] for i in range(len(OBSERVATIONS))]).astype(np.float32)
        elif self.obs_type == "img":
            obs = np.zeros((1,NUM_E_FEATURES, NUM_EDGES), dtype=np.float32)
            for i in range(NUM_EDGES):
                if 1 <= i < NUM_EDGES - 1:
                    obs[0][0][i] = self.state[OBSERVATIONS.index(f"num_vehs_edge_{i}_in_PTL")]
                    obs[0][1][i] = self.state[OBSERVATIONS.index(f"mean_speed_edge_{i}_in_PTL")]
                    obs[0][2][i] = self.state[OBSERVATIONS.index(f"std_speed_edge_{i}_in_PTL")]
                obs[0][3][i] = self.state[OBSERVATIONS.index(f"num_vehs_edge_{i}_no_PTL")]
                obs[0][4][i] = self.state[OBSERVATIONS.index(f"mean_speed_edge_{i}_no_PTL")]
                obs[0][5][i] = self.state[OBSERVATIONS.index(f"std_speed_edge_{i}_no_PTL")]
            return obs.astype(np.float32)

    def reset(self, seed=None, options=None, ):
        if self.model:
            final_log = log_features(self.policy_name + exp_name + "_av" + str(self.av_rate) + ".xml", self.timestep,
                         self.act_rate)
            mean_pass_delay = final_log["mean_pass_delay"]
            if mean_pass_delay < self.best_mean_pass_delay:
                os.remove("agents/" + self.agent_name + "_" + str(round(self.best_mean_pass_delay,0)))
                self.best_mean_pass_delay = mean_pass_delay
                self.model.save("agents/" + self.agent_name + "_" + str(round(mean_pass_delay,0)))

        # check if a traci instance is already running
        try:
            traci.close()
        except:
            pass
        self.timestep = 0
        _, _, self.av_rate = init_simulation((self.policy_name, self.sumoCfg))
        self.av_rate = float(self.av_rate[2:])
        self.state = [0] * len(OBSERVATIONS)

        for i in range(self.act_rate//1000):
            self.actions[5](self)
        self.last_action = 5
        return self.observation(), {}

    def step(self, action):
        if self.action_space_tag == "alter":
            if action == 0 and self.last_action > 1:
                self.last_action -= 1
            elif action == 2 and self.last_action < 5:
                self.last_action += 1
            action = self.last_action

        reward, done, log_msg = action_wrapper(self, "StaticNumPassFL_" + str(action + 1))
        self.log += f"Time: {self.timestep - self.act_rate} Action: {action + 1}, Reward: {reward}\n"
        if self.log_wandb and log_msg:
            log_msg["MinNumPass"] = action + 1
            log_msg["Reward"] = reward
            wandb.log(log_msg)
        obs = self.observation()
        return obs, reward, done, False, {}

    def render(self, mode=None):
        print(self.log)
        self.log = ''

class SaveOnResetCallback(BaseCallback):
    def __init__(self, env, verbose=1):
        super(SaveOnResetCallback, self).__init__(verbose)
        self.env = env.env.env

    def _on_step(self) -> bool:
        self.env.model = self.model
        return True


def train_agent(cfg):
    sumoCfg, feat_type, act_rate,agent_type, action_space_tag = cfg
    policy_name = agent_type + "_" + feat_type + "_ACT_RATE_" + str(act_rate)+ "_" + action_space_tag
    # Register the environment with Gym
    gym.envs.registration.register(
        id='LeftLaneENV-v0',
        entry_point=LeftLaneENV,
    )

    env = gym.make('LeftLaneENV-v0', policy_name=policy_name, sumoCfg=sumoCfg, features_type=feat_type,
                   act_rate=act_rate, action_space_tag=action_space_tag)
    if agent_type == "DQN":
        model = DQN("MlpPolicy", env, verbose=1, learning_starts=1, target_update_interval=(3600*13)//act_rate)
    elif agent_type == "DQN_CNN":
        model = DQN("CnnPolicy", env, verbose=1)
    elif agent_type == "PPO":
        model = PPO("MlpPolicy", env, verbose=1)
    elif agent_type == "PPO_CNN":
        model = PPO("CnnPolicy", env, verbose=1)
    elif agent_type == "A2C":
        model = A2C("MlpPolicy", env, verbose=1)
    elif agent_type == "A2C_CNN":
        model = A2C("CnnPolicy", env, verbose=1)

    os.makedirs("agents", exist_ok=True)
    save_callback = SaveOnResetCallback(env)
    model.learn(total_timesteps=num_it, callback=save_callback)

    env.close()


if __name__ == '__main__':

    sumoCfgs = [f"../cfg_files_LeftCompDaily/LeftCompDaily_av{r}.sumocfg" for r in [0.2, 0.4, 0.6, 0.8, 0.5]]
    # sumoCfgs = [f"../cfg_files_LeftCompDaily/LeftCompDaily_av{r}.sumocfg" for r in [0.5]]
    # agent_types = ["DQN_CNN", "PPO_CNN", "A2C_CNN","DQN", "PPO", "A2C", ]
    agent_types = ["DQN"]
    # feat_types = ["E_FEATURES_TU","LOG_FEATURES","E_FEATURES"]
    feat_types = ["MS_EPTL"]
    ACTION_SPACE_TAGS = ["alter", "direct"]
    act_rates = [100, 300]
    cfgs = [(sumoCfg, feat_type, act_rate, agent_type, action_space_tag) for sumoCfg in sumoCfgs for feat_type in feat_types
            for act_rate in act_rates for agent_type in agent_types for action_space_tag in ACTION_SPACE_TAGS]
    cfgs_clean = [cfg for cfg in cfgs if not(cfg[1] == "LOG_FEATURES" and cfg[3].endswith("CNN"))]
    print("num cfgs", len(cfgs_clean))
    with Pool(len(cfgs_clean)) as pool:
        tqdm(pool.map(train_agent, cfgs_clean), total=len(sumoCfgs))