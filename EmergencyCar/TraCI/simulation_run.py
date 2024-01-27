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
SIM_DURATION = 7200
NUM_PROCESSES = 1
NUM_REPS = 1
EMERGENCY_PROB = 0.003
POLICIES = ["ClearFront", "Nothing"]

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
    exp_output_name = "results_reps_long/"+policy_name+"_"+".".join(sumoCfg.split("/")[-1].split(".")[:-1])+".xml"

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
    parallel_simulation(args)