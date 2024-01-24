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
NUM_PROCESSES = 11
NUM_REPS = 1
EMERGENCY_PROB = 0.003

# Traffic parameters
AV_PROB = None  # testing many AV probabilities

sumoCfg = fr"..\{exp_name}.sumocfg"
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
sumoCmd = [sumoBinary, "-c", sumoCfg, "--tripinfo-output"]


def simulate(arg):
    av_rate, policy_name, major_flow, seed = arg
    randomizer = np.random.default_rng(seed=int(seed))
    for i in range(NUM_REPS):
        sumoCmdRep = sumoCmd.copy()
        rep_output_path = f"results_reps_long/{policy_name}_flow_{major_flow}_av_rate_{av_rate}_rep_{i}.xml"
        sumoCmdRep += [rep_output_path, "--seed", seed]
        traci.start(sumoCmdRep)
        step = 0
        while traci.simulation.getMinExpectedNumber() > 0:
            handle_step(step, av_rate, EMERGENCY_PROB, policy_name, randomizer)
            traci.simulationStep(step)
            step += 1
        traci.close()


def parallel_simulation(av_rates, major_flow, policy_name,seed):
    args = [(av_rate, policy_name,major_flow, seed) for av_rate in av_rates]
    with Pool(NUM_PROCESSES) as pool:  # 10 processes
        results = list(tqdm(pool.imap(simulate, args), total=len(av_rates)))


if __name__ == "__main__":
    for major_flow in [1000, 2000, 3000, 4000, 5000]:
        set_sumo_simulation(major_flow, SIM_DURATION)
        av_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        seed = str(np.random.randint(0, 1000000))
        for policy_name in ["ClearFront", "Nothing"]:
            parallel_simulation(av_rates,major_flow, policy_name=policy_name, seed=seed)
            parse_output_files(av_rates, policy_name=policy_name, num_reps=NUM_REPS, flow=major_flow)