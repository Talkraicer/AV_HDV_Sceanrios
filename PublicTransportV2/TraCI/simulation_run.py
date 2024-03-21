import os
import sys
import numpy as np
import pandas as pd



from tqdm import tqdm
from multiprocessing import Pool
from utils import *
import traci

GUI = True

# SIM parameters
SIM_DURATION = 86400
NUM_PROCESSES = 1
POLICIES = ["Nothing"]
AV_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1.0]

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
    exp_output_name = "results_reps/"+policy_name+"_"+".".join(sumoCfg.split("/")[-1].split(".")[:-1])+".xml"

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
    for sumoCfg in os.listdir("../cfg_files"):
        if sumoCfg.endswith(".sumocfg"):
            sumoCfgPath = f"../cfg_files/{sumoCfg}"
            sumoCfgPaths.append(sumoCfgPath)
    args = []
    for policy in POLICIES:
        for sumoCfg in sumoCfgPaths:
            args.append((policy, sumoCfg))
    parallel_simulation(args)
    policies_clean = [policy for policy in POLICIES if policy != "Nothing"]
    # parse_all_pairwise(policies_clean, "Nothing", FLOWS, AV_rates,BUS_PROB)
    # parse_all_pairwise(policies_clean, "NothingDL", FLOWS, AV_rates,BUS_PROB)
    # convert_all_flows_to_av_rates(policies_clean, "Nothing", FLOWS, AV_rates,BUS_PROB)
    # convert_all_flows_to_av_rates(policies_clean, "NothingDL", FLOWS, AV_rates,BUS_PROB)


