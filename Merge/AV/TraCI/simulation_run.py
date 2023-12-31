import os
import sys
from random import random
import numpy as np
import pandas as pd
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import traci
import traci.constants as tc
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool

GUI = False

NUM_REPS = 10
DETECT_MERGING_LOC = 40
SLOW_LOC = 35
SLOW_LEN = 15

sumoCfg = r"..\merge.sumocfg"
sumoBinary = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe" if GUI else \
    r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe"
sumoCmd = [sumoBinary, "-c", sumoCfg]


def handle_step(t, av_prob, detect_merging_loc, slow_loc, slow_len, desired_slow_speed):
    vehIDs = traci.vehicle.getIDList()
    merging = False
    for vehID in vehIDs:
        if traci.vehicle.getTypeID(vehID) == "DEFAULT_VEHTYPE":
            traci.vehicle.setType(vehID, "AV" if random() < av_prob else "HD")
        speed, lane, lanePos = traci.vehicle.getSpeed(vehID), traci.vehicle.getLaneID(vehID), \
            traci.vehicle.getLanePosition(vehID)
        if lane == "E2_0" and lanePos > detect_merging_loc:
            merging = True
        # print(vehID, ": ", lane, " SPEED:", speed, " LANEPOS:", lanePos, " TYPE:", traci.vehicle.getTypeID(vehID))
    if merging:
        for vehID in vehIDs:
            if traci.vehicle.getLaneID(vehID) == "E0_0" and traci.vehicle.getTypeID(vehID) == "AV" and \
                    slow_loc < traci.vehicle.getLanePosition(vehID) < slow_loc + slow_len:
                # Slow down the vehicle to specific period of time
                traci.vehicle.slowDown(vehID, desired_slow_speed, 1)

    return True



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
    df.to_csv(f"results_{NUM_REPS}_500Major.csv")

if __name__ == "__main__":
    desired_slow_speeds = np.arange(0, 10, 1)
    parallel_simulation(desired_slow_speeds)


