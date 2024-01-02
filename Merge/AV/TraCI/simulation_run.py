import os
import sys
import numpy as np
import pandas as pd
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import traci
import traci.constants as tc
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool
from utils import *

GUI = False

# SIM parameters
NUM_REPS = 10
SIM_DURATION = 600

# Traffic parameters
MAJOR_FLOW = 1500
AV_PROB = None # testing many AV probabilities

# Controllable parameters
DETECT_MERGING_LOC = 40
SLOW_LOC = 35
SLOW_LEN = 15
DESIRED_SLOW_SPEED = None # testing many desired slow speeds

sumoCfg = r"..\merge.sumocfg"
sumoBinary = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe" if GUI else \
    r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe"
sumoCmd = [sumoBinary, "-c", sumoCfg]



def simulate(desired_slow_speed):
    speed_results_avg = []
    speed_results_std = []
    for av_prob in np.arange(0, 1.1, 0.1):
        speed_prob_results = []
        for _ in range(NUM_REPS):
            traci.start(sumoCmd)
            step = 0
            while traci.simulation.getMinExpectedNumber() > 0:
                handle_step(step, av_prob, DETECT_MERGING_LOC, SLOW_LOC, SLOW_LEN, desired_slow_speed)
                traci.simulationStep(step)
                step += 1
            speed_prob_results.append(step)
            traci.close(False)
        speed_results_avg.append(np.mean(speed_prob_results))
        speed_results_std.append(np.std(speed_prob_results))
    return f"slow_speed_{desired_slow_speed}_avg", f"slow_speed_{desired_slow_speed}_std", speed_results_avg, speed_results_std


def parallel_simulation(desired_slow_speeds):
    with Pool(10) as pool:  # 10 processes
        results = list(tqdm(pool.imap(simulate, desired_slow_speeds), total=len(desired_slow_speeds)))

    # Organize results into a DataFrame
    df = pd.DataFrame(index=np.arange(0, 1.1, 0.1))
    for result in results:
        avg_key, std_key, avg_values, std_values = result
        df[avg_key] = avg_values
        df[std_key] = std_values
    df.index = np.arange(0, 1.1, 0.1)
    df.to_csv(f"results_{NUM_REPS}_{MAJOR_FLOW}Major_{SIM_DURATION}Duration.csv")

if __name__ == "__main__":
    desired_slow_speeds = np.arange(0, 10, 1)
    set_sumo_simulation(MAJOR_FLOW, SIM_DURATION)
    # record_sumo_simulation_to_gif(MAJOR_FLOW)
    parallel_simulation(desired_slow_speeds)


