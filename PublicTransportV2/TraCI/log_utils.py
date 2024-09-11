from xml.etree import ElementTree as ET
import pandas as pd
import numpy as np
import wandb
import os
import traci
from results_utils import output_file_to_df, calc_stats_metric
from utils import exp_name
results_reps_folder = "results_reps"
NUM_EDGES = 8

def log_features(output_file,t, log_rate):
    # calc all vehicles speed in the road
    vehIDs = traci.vehicle.getIDList()
    mean_speed = np.mean([traci.vehicle.getSpeed(vehID) for vehID in vehIDs])
    mean_speed_in_end_PTL = traci.lane.getLastStepMeanSpeed("E6_3")
    num_vehs_in_PTL = sum(
        [traci.lane.getLastStepVehicleNumber(l) for l in traci.lane.getIDList() if len(traci.lane.getAllowed(l)) > 0])
    num_total_vehs = len(vehIDs)
    num_hdv_in_end_PTL = sum([traci.lane.getLastStepVehicleNumber(f"E6_{i}") for i in range(3)])

    num_allowed_vehs_PTL = sum([1 for vehID in vehIDs if traci.vehicle.getVehicleClass(vehID) in ["private","bus"]])

    PTL_speeds = []
    for e_idx in range(NUM_EDGES):
        edge = "E" + str(e_idx)
        if 1 <= e_idx < NUM_EDGES - 1:
            num_lanes = traci.edge.getLaneNumber(edge)
            PTL_idx = num_lanes - 1
            PTL_speeds += [traci.vehicle.getSpeed(vehID) for vehID in
                        traci.lane.getLastStepVehicleIDs(f"{edge}_{PTL_idx}")]

    mean_speed_in_PTL = np.mean(PTL_speeds)

    # calc arrived passengers mean total delay
    output_file = f"{results_reps_folder}/{output_file}"
    with open(output_file, "r") as f:
        txt = f.read()
    start_arriving = "<tripinfo " in txt
    mean_pass_delay = 0
    if start_arriving:
        # fix end of <tripinfo> tag
        with open(output_file, "a+") as f:
            f.write("</tripinfos>")
        df = output_file_to_df(output_file)
        total_delay = calc_stats_metric(df, "totalDelay", diff=False)
        mean_pass_delay = total_delay.loc["avg_totalDelay", "Passenger"]

        df_timestamp = df[df["arrivalTime"] > t-log_rate]
        total_delay_timestamp = calc_stats_metric(df_timestamp, "totalDelay", diff=False)
        mean_pass_delay_timestamp = total_delay_timestamp.loc["avg_totalDelay", "Passenger"]

        # remove the <tripinfo> tag
        with open(output_file, "r") as f:
            lines = f.readlines()
        with open(output_file, "w") as f:
            f.writelines(lines[:-1])

        log_msg = {"num_vehs_in_PTL": num_vehs_in_PTL, "num_total_vehs": num_total_vehs,
                   "num_hdv_in_end_PTL": num_hdv_in_end_PTL, "mean_speed": mean_speed,
                   "mean_speed_in_end_PTL": mean_speed_in_end_PTL, "mean_pass_delay": mean_pass_delay,
                   "num_allowed_vehs_PTL": num_allowed_vehs_PTL, "mean_speed_in_PTL": mean_speed_in_PTL,
                   "mean_pass_delay_timestamp": mean_pass_delay_timestamp
                   }
        return log_msg

def init_wandb_logger(policy_name,av_rate,delete_older=False):
    proj_name = exp_name + "_" + str(av_rate)

    api = wandb.Api()
    username = api.default_entity

    projects = api.projects(username)
    if delete_older and proj_name in [proj.name for proj in projects]:
        # Retrieve the run ID (you can also manually set this if you know the ID)
        runs = api.runs(f"{username}/{proj_name}")

        # Delete the run if it exists
        deleted = False
        for run in runs:
            if run.name == policy_name:
                run = api.run(f"{username}/{proj_name}/{run.id}")
                run.delete()
                deleted = True
                break
        if not deleted:
            print(f"Run {policy_name} not found")
    wandb.init(project=proj_name, name=policy_name)
