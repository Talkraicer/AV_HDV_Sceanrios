import os
import sys
import numpy as np
import pandas as pd

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
from tqdm import tqdm
from multiprocessing import Pool
from utils import *
import traci

GUI = False

# SIM parameters
NUM_REPS = 10
SIM_DURATION = 400

# Traffic parameters
MAJOR_FLOW = 1000
AV_PROB = None  # testing many AV probabilities

sumoCfg = fr"..\{exp_name}.sumocfg"
sumoBinary = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe" if GUI else \
    r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe"
sumoCmd = [sumoBinary, "-c", sumoCfg, "--tripinfo-output"]


def simulate(arg):
    av_rate, policy_name, major_flow = arg
    for i in range(NUM_REPS):
        sumoCmdRep = sumoCmd.copy()
        rep_output_path = f"results_reps/{policy_name}_flow_{major_flow}_av_rate_{av_rate}_rep_{i}.xml"
        sumoCmdRep += [rep_output_path, "--seed", str(np.random.randint(10000))]
        traci.start(sumoCmdRep)
        step = 0
        while traci.simulation.getMinExpectedNumber() > 0:
            handle_step(step, av_rate, policy_name)
            traci.simulationStep(step)
            step += 1
        traci.close()


def parallel_simulation(av_rates, major_flow, policy_name):
    args = [(av_rate, policy_name,major_flow) for av_rate in av_rates]
    with Pool(11) as pool:  # 10 processes
        results = list(tqdm(pool.imap(simulate, args), total=len(av_rates)))


if __name__ == "__main__":
    for major_flow in [2000, 3000, 4000, 5000]:
        set_sumo_simulation(major_flow, SIM_DURATION)
        av_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for policy_name in ["ClearLeft", "ClearLeftWhenEmergency", "ClearFront", "Nothing"]:
            parallel_simulation(av_rates,major_flow, policy_name=policy_name)
            parse_output_files(av_rates, policy_name=policy_name, num_reps=NUM_REPS, flow=major_flow)
