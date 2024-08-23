import os
import sys
import numpy as np
import pandas as pd

from tqdm import tqdm
from multiprocessing import Pool
from joblib import parallel_backend

from utils import *
import traci
import optuna
import warnings
warnings.filterwarnings("ignore")

GUI = False

# SIM parameters
SIM_DURATION = 86400
NUM_PROCESSES = 70
AV_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
EXP_NAME_TAG = "LeftCompScenarios"

# POLICIES = ["Plus", "Control mean_speed_in_end_PTL", "Nothing", "StaticNumPassFL"]
POLICIES = ["Control mean_speed_in_end_PTL"]
# CONTROL_SPEED_RANGES = [(13,20),(16,21),(16,20),(15,22),(14,22)]
# CONTROL_SPEED_RANGES = [(10,18),(10,20),(14,20),(16,22),(8,15),(8,20),(8,18)]
CONTROL_SPEED_RANGES = [(12,20)]

# parameters for StaticNumPass
MIN_NUM_PASS = [1, 2, 3, 4, 5]

# parameters for DisallowBack
STOP_FROM_RANGE = [800, 1000, 1200]
STOP_TO_RANGE = [0, 100, 200]

# parameters for EnterClear
EnterClearRange = [200, 300, 400, 500]

# parameters for FastLane
Features_Dist = ["1200"]
Max_AVS = ["5", "10", "15", "20"]
Max_Buses = ["0", "1", "2"]

if GUI:
    NUM_PROCESSES = 1

if 'SUMO_HOME' in os.environ:
    sumo_path = os.environ['SUMO_HOME']
    sys.path.append(os.path.join(sumo_path, 'tools'))
    # check operational system - if it is windows, use sumo.exe if linux, use sumo
    if os.name == 'nt':
        sumoBinary = os.path.join(sumo_path, 'bin', 'sumo-gui.exe') if GUI else \
            os.path.join(sumo_path, 'bin', 'sumo.exe')
    else:
        sumoBinary = os.path.join(sumo_path, 'bin', 'sumo-gui') if GUI else \
            os.path.join(sumo_path, 'bin', 'sumo')
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


def init_simulation(arg):
    policy_name, sumoCfg = arg
    sumoCmd = [sumoBinary, "-c", sumoCfg, "--tripinfo-output"]
    exp_output_name = "results_reps/" + policy_name + ".".join(sumoCfg.split("/")[-1].split(".")[:-1]) + ".xml"
    av_rate = ".".join(sumoCfg.split("/")[-1].split(".")[:-1]).split("_")[-1]
    sumoCmd.append(exp_output_name)
    traci.start(sumoCmd)
    return policy_name, sumoCfg, av_rate


def simulate(arg, log_wandb=True):
    policy_name, sumoCfg, av_rate = init_simulation(arg)
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        if log_wandb:
            handle_step(step, policy_name, av_rate)
        else:
            handle_step(step, policy_name, av_rate, log_rate=0)
        traci.simulationStep(step)
        step += 1
    traci.close()


def parallel_simulation(args):
    with Pool(NUM_PROCESSES) as pool:  # 10 processes
        results = list(tqdm(pool.imap(simulate, args), total=len(args)))


def optuna_simulation(sumoCfgPath):
    study = optuna.create_study(direction='minimize')

    def optuna_objective(trial):
        min_speed = trial.suggest_float('min_speed', 8, 16)
        max_speed = trial.suggest_float('max_speed', min_speed + 2, 22)
        policy_name = f"Control mean_speed_in_end_PTL {min_speed} {max_speed}"
        simulate((policy_name, sumoCfgPath), log_wandb=False)
        av_rate = ".".join(sumoCfgPath.split("/")[-1].split(".")[:-1]).split("_")[-1]
        output_file = "results_reps/" + policy_name + exp_name + "_" + str(av_rate) + ".xml"
        df = output_file_to_df(output_file)
        total_delay = calc_stats_metric(df, "totalDelay", diff=False)
        mean_pass_delay = total_delay.loc["avg_totalDelay", "Passenger"]
        return mean_pass_delay

    study.optimize(optuna_objective, n_trials=100, n_jobs=1, show_progress_bar=True)
    with open("optuna_results.txt", "a+") as f:
        f.write(f"SumoCfg: {sumoCfg}\n")
        f.write(f"Best value: {study.best_value}\n")
        f.write(f"Best params: {study.best_params}\n")
        f.write("\n")


def simulate_policies(sumoCfgPaths):
    args = []
    for policy in POLICIES:
        policy_name = policy
        for sumoCfg in sumoCfgPaths:
            if policy.startswith("DisallowBack"):
                for stop_from in STOP_FROM_RANGE:
                    for stop_to in STOP_TO_RANGE:
                        policy_name = f"{policy}_{stop_from}_{stop_to}"
                        args.append((policy_name, sumoCfg))
            elif policy.startswith("FastLane"):
                for feature_dist in Features_Dist:
                    for max_avs in Max_AVS:
                        for max_buses in Max_Buses:
                            policy_name = f"{policy}_{feature_dist}_{max_avs}_{max_buses}"
                            args.append((policy_name, sumoCfg))
            elif policy.startswith("EnterClear"):
                for enter_clear in EnterClearRange:
                    policy_name = f"{policy}_{enter_clear}"
                    args.append((policy_name, sumoCfg))
            elif policy.startswith("StaticNumPass") or policy.startswith("Plus"):
                for min_num_pass in MIN_NUM_PASS:
                    policy_name = f"{policy}_{min_num_pass}"
                    args.append((policy_name, sumoCfg))
            elif policy.startswith("Control"):
                for min_speed, max_speed in CONTROL_SPEED_RANGES:
                    policy_name = f"{policy} {min_speed} {max_speed}"
                    args.append((policy_name, sumoCfg))
            else:
                args.append((policy_name, sumoCfg))
    parallel_simulation(args)


def parse_results():
    policies = []
    for policy in POLICIES:
        if policy.startswith("StaticNumPass") or policy.startswith("Plus"):
            for min_num_pass in MIN_NUM_PASS:
                policies.append(f"{policy}_{min_num_pass}")
        elif policy.startswith("Control"):
            for min_speed, max_speed in CONTROL_SPEED_RANGES:
                policies.append(f"{policy} {min_speed} {max_speed}")
        else:
            policies.append(policy)
        policy_names = [f"{policy}_{enter_clear}" for policy in POLICIES if policy.startswith("EnterClear")
                        for enter_clear in EnterClearRange]
        policy_names = ["Nothing"]
        parse_all_output_files(AV_rates, 1, policies)
        if "Nothing" in policies:
            policies.remove("Nothing")
        # parse_all_pairwise(policies, AV_rates)
        # policy_names = [f"{policy}_{stop_from}_{stop_to}" for policy in POLICIES if policy.startswith("DisallowBack")
        #                 for stop_from in STOP_FROM_RANGE for stop_to in STOP_TO_RANGE]
        # policy_names += [f"{policy}_{feature_dist}_{max_avs}_{max_buses}" for policy in POLICIES if policy.startswith("FastLane")]
        # policy_names += [policy for policy in POLICIES if not policy.startswith("DisallowBack") and not policy.startswith("FastLane")]
        # create_all_results_tables(AV_rates,policy_names)
        # parse_all_output_files(AV_rates, 1, policy_names)
        # parse_all_pairwise(policy_names, AV_rates)


if __name__ == "__main__":
    sumoCfgPaths = []
    for sumoCfg in os.listdir(f"../cfg_files_{EXP_NAME_TAG}"):
        if sumoCfg.endswith(".sumocfg"):
            # if ("0.1" in sumoCfg or "0.3" in sumoCfg):
            sumoCfgPath = f"../cfg_files_{EXP_NAME_TAG}/{sumoCfg}"
            sumoCfgPaths.append(sumoCfgPath)

    print("Number of sumoCfg files: ", len(sumoCfgPaths))
    if GUI:
        sumoCfgPaths = [sumoCfgPaths[3]]
    # optuna_simulation([f"../cfg_files_{EXP_NAME_TAG}/LeftCompDaily_av0.5.sumocfg"])
    # with Pool(len(sumoCfgPaths)) as p:
        # p.map(optuna_simulation, sumoCfgPaths)
    simulate_policies(sumoCfgPaths)
