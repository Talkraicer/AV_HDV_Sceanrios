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
POLICIES = ["SlowDown","Nothing"]
DIST_SLOW_RANGE = [200, 300, 400, 500, 600, 700, 800]
DIST_FAST_RANGE = [100,150,200]
SLOW_RATE_RANGE = [0.6, 0.8]
STOPPING_LANES = [0,2]

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
    policy_name, sumoCfg,stopping_lane, dist_slow, dist_fast, slow_rate = arg
    sumoCmd = [sumoBinary, "-c", sumoCfg, "--tripinfo-output"]
    policy_name_output = policy_name+"_dist_slow_"+str(dist_slow)+"_dist_fast_"+str(dist_fast)+\
                         "_slow_rate_"+str(slow_rate)+"_stopping_lane_"+str(stopping_lane)
    exp_output_name = "results_reps/"+policy_name_output+"_"+".".join(sumoCfg.split("/")[-1].split(".")[:-1])+".xml"

    sumoCmd.append(exp_output_name)
    traci.start(sumoCmd)
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        handle_step(t=step, policy_name=policy_name, stopping_lane=stopping_lane,
                    dist_slow=dist_slow, dist_fast=dist_fast, slow_rate=slow_rate)
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
    for policy_name in POLICIES:
        for stopping_lane in STOPPING_LANES:
            if policy_name == "SlowDown":
                for dist_slow in DIST_SLOW_RANGE:
                    for dist_fast in DIST_FAST_RANGE:
                        for slow_rate in SLOW_RATE_RANGE:
                            for sumoCfg in sumoCfgPaths:
                                args.append((policy_name, sumoCfg,stopping_lane, dist_slow, dist_fast, slow_rate))
            else:
                for sumoCfg in sumoCfgPaths:
                    args.append((policy_name, sumoCfg,stopping_lane, 0, 0, 0))
    parallel_simulation(args)