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
POLICIES = ["ClearFront500","ClearFront","ClearFront100","Nothing"]
FLOWS = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
AV_PROB = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Traffic parameters
AV_PROB = None  # testing many AV probabilities

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

    args = [(policy_name, sumoCfgPath) for policy_name in POLICIES for sumoCfgPath in sumoCfgPaths]
    if "Nothing" in POLICIES:
        for sumoCfg in os.listdir("../cfg_files"):
            if sumoCfg.endswith(".sumocfg"):
                sumoCfgPath = f"../cfg_files_DL/{sumoCfg}"
                args.append(("NothingDL", sumoCfgPath))
    parallel_simulation(args)
    parse_all_pairwise(policies=POLICIES,policy_name2="Nothing",flows=FLOWS,av_rates=AV_PROB)



