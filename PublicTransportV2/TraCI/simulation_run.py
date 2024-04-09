import os
import sys
import numpy as np
import pandas as pd



from tqdm import tqdm
from multiprocessing import Pool
from utils import *
import traci

GUI = False

# SIM parameters
SIM_DURATION = 86400
NUM_PROCESSES = 70
POLICIES = ["DisallowBack"]
STOP_FROM_RANGE = [300,400,500,600,700,800,900,1000,1100,1200]
STOP_TO_RANGE = [0,100,200]
AV_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1.0]

if GUI:
    NUM_PROCESSES = 1

# Traffic parameters

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


def simulate(arg):
    policy_name, sumoCfg = arg
    sumoCmd = [sumoBinary, "-c", sumoCfg, "--tripinfo-output"]
    exp_output_name = "results_reps/"+policy_name+".".join(sumoCfg.split("/")[-1].split(".")[:-1])+".xml"
    sumoCmd.append(exp_output_name)
    traci.start(sumoCmd)
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        handle_step(step, policy_name)
        traci.simulationStep(step)
        step += 1
    traci.close()


def parallel_simulation(args):
    with Pool(NUM_PROCESSES) as pool:  # 10 processes
        results = list(tqdm(pool.imap(simulate, args), total=len(args)))


if __name__ == "__main__":
    sumoCfgPaths = []
    for sumoCfg in os.listdir("../cfg_files_Bay"):
        if sumoCfg.endswith(".sumocfg"):
            sumoCfgPath = f"../cfg_files_Bay/{sumoCfg}"
            sumoCfgPaths.append(sumoCfgPath)
    if GUI:
        sumoCfgPaths = [sumoCfgPaths[5]]
    args = []
    for policy in POLICIES:
        policy_name = policy
        for sumoCfg in sumoCfgPaths:
            if policy == "DisallowBack":
                for stop_from in STOP_FROM_RANGE:
                    for stop_to in STOP_TO_RANGE:
                        policy_name = f"{policy}_{stop_from}_{stop_to}"
                        args.append((policy_name, sumoCfg))
            else:
                args.append((policy_name, sumoCfg))
    parallel_simulation(args)

    parse_output_files(AV_rates, 1, "Nothing")
    parse_all_pairwise(POLICIES, AV_rates)



