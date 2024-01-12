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
SIM_DURATION = 300

# Traffic parameters
MAJOR_FLOW = 1000
AV_PROB = None  # testing many AV probabilities

sumoCfg = fr"..\{exp_name}.sumocfg"
sumoBinary = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe" if GUI else \
    r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe"
sumoCmd = [sumoBinary, "-c", sumoCfg, "--tripinfo-output"]


def simulate(arg):
    av_rate, policy_name = arg
    steps_list = []
    emergency_steps_list = []
    arrival_emergency_list = []
    for i in range(NUM_REPS):
        sumoCmdRep = sumoCmd
        rep_output_path = f"results_reps/av_rate_{av_rate}_rep_{i}.xml"
        sumoCmdRep += [rep_output_path, "--seed", str(np.random.randint(10000))]
        traci.start(sumoCmdRep)
        step = 0
        emergency_steps = 0
        arrival_emergency = 0
        while traci.simulation.getMinExpectedNumber() > 0:
            if handle_step(step, av_rate, policy_name):
                if emergency_steps == 0:
                    arrival_emergency = step
                emergency_steps += 1
            traci.simulationStep(step)
            step += 1

        steps_list.append(step)
        emergency_steps_list.append(emergency_steps)
        arrival_emergency_list.append(arrival_emergency)
        traci.close(False)
    return np.average(emergency_steps_list), np.std(emergency_steps_list), np.average(steps_list), np.std(steps_list), \
        np.average(arrival_emergency_list), np.std(arrival_emergency_list)


def parallel_simulation(av_rates, policy_name="Nothing"):
    args = [(av_rate, policy_name) for av_rate in av_rates]
    with Pool(1) as pool:  # 10 processes
        results = list(tqdm(pool.imap(simulate, args), total=len(av_rates)))
    df = pd.DataFrame(results, columns=['emergency_steps_avg', 'emergency_steps_std', 'sim_steps_avg',
                                        'sim_steps_std', 'arrival_emergency_avg', 'arrival_emergency_std'])
    df['av_rate'] = av_rates
    df.set_index('av_rate', inplace=True)
    df.to_csv(f"results_csvs/results_{exp_name}_{NUM_REPS}_{MAJOR_FLOW}Major_{SIM_DURATION}Duration_{policy_name}.csv")


if __name__ == "__main__":
    set_sumo_simulation(MAJOR_FLOW, SIM_DURATION)
    av_rates = [0.5]
    for policy_name in ["ClearLeft"]:
        # record_sumo_simulation_to_gif(MAJOR_FLOW,policy_name=policy_name, record_duration=SIM_DURATION, av_prob=0.5)
        parallel_simulation(av_rates, policy_name=policy_name)
