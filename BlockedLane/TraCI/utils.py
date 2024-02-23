import os
import traci
import imageio
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

exp_name = "BlockedLane"
NUM_PROCESSES = 70
GUI = True
sumoCfg = fr"../{exp_name}.sumocfg"
results_folder = "results_csvs"
results_reps_folder = "results_reps"
metrics = ["duration", "departDelay", "speed", "timeLoss", "totalDelay"]

chosen_avs = {}

def is_leader_AV_before_stopper(veh_id):
    # check if the vehicle is the leader of the AVs before the stopper
    if traci.vehicle.getTypeID(veh_id) == "AV":
        dist = 0
        leader = traci.vehicle.getLeader(veh_id,0)
        while leader is not None:
            frontVehID, dist_leader = leader
            dist += dist_leader
            if frontVehID == "stopping" and traci.vehicle.getLaneID(veh_id) == traci.vehicle.getLaneID(frontVehID):
                return True, dist
            leader = traci.vehicle.getLeader(frontVehID,0)
    return False, 0

def get_inside_farest(possible_avs, dist_slow, dist_fast):
    # get the vehicle that is inside the slow and fast distances and is the closest to the dist slow
    min_dist_to_slow = float("inf")
    arg_min_dist_to_slow = None
    for veh_id, dist in possible_avs.items():
        if dist_fast < dist < dist_slow:
            if dist_slow - dist < min_dist_to_slow:
                min_dist_to_slow = dist_slow - dist
                arg_min_dist_to_slow = veh_id, dist
    return arg_min_dist_to_slow

def get_outside_closest(possible_avs, dist_slow, dist_fast):
    # get the vehicle that is outside the slow and fast distances and is the closest to the dist slow
    min_dist_to_fast = float("inf")
    arg_min_dist_to_fast = None
    for veh_id, dist in possible_avs.items():
        if dist > dist_slow:
            if dist - dist_slow < min_dist_to_fast:
                min_dist_to_fast = dist - dist_slow
                arg_min_dist_to_fast = veh_id,dist
    return arg_min_dist_to_fast

def handle_step(t, policy_name, dist_slow=0, dist_fast=0, slow_rate=0, stopping_lane=1):
    global chosen_avs
    if t == 0:
        traci.vehicle.add(vehID="stopping", routeID="r_0")
    if "stopping" not in traci.vehicle.getIDList():
        return
    if t >= 35:
        traci.vehicle.setSpeed("stopping", 0)
        traci.vehicle.changeLane("stopping", stopping_lane, 0)
        stopping_lane_id = traci.vehicle.getLaneID("stopping")
        slow_speed = traci.lane.getMaxSpeed(stopping_lane_id) * slow_rate
        if traci.vehicle.getSpeed("stopping") == 0:
            if policy_name == "SlowDown":
                veh_ids = traci.vehicle.getIDList()
                possible_avs = {}
                for veh_id in veh_ids:
                    is_leader, dist = is_leader_AV_before_stopper(veh_id)
                    if is_leader:
                        possible_avs[veh_id] = dist
                if len(possible_avs) > 0 and len(chosen_avs) == 0:
                    inside_farest = get_inside_farest(possible_avs, dist_slow, dist_fast)
                    if inside_farest is not None:
                        chosen_avs[inside_farest[0]] = inside_farest[1]
                new_chosen_avs = chosen_avs.copy()
                for veh_id in chosen_avs.keys():
                    # update the distance of the chosen avs
                    if veh_id not in possible_avs:
                        new_chosen_avs.pop(veh_id)
                        continue
                    dist = possible_avs[veh_id]
                    chosen_avs[veh_id] = dist
                    chosen_speed = traci.vehicle.getSpeed(veh_id)
                    if chosen_speed == 0:
                        new_chosen_avs.pop(veh_id)

                    # make sure the vehicle is in the correct lane and color
                    traci.vehicle.changeLane(veh_id, stopping_lane, 0)
                    traci.vehicle.setColor(veh_id, (128, 0, 128))  # purple

                    if dist < dist_fast:
                        # print("Releasing vehicle from slow down: veh_id = ", veh_id, "dist = ", dist, "dist_fast = ", dist_fast)
                        traci.vehicle.setSpeed(veh_id, -1)
                        new_chosen_avs.pop(veh_id)
                        outside_closest = get_outside_closest(possible_avs, dist_slow, dist_fast)
                        if outside_closest is not None:
                            new_chosen_avs[outside_closest[0]] = outside_closest[1]
                        inside_closest = get_inside_farest(possible_avs, dist_slow, dist_fast)
                        if inside_closest is not None:
                            new_chosen_avs[inside_closest[0]] = inside_closest[1]
                    elif dist < dist_slow:
                        if chosen_speed > slow_speed:
                            # print("Slowing down vehicle: veh_id = ", veh_id, "dist = ", dist,
                            #       "from speed = ", traci.vehicle.getSpeed(veh_id), "to speed = ", slow_speed)
                            traci.vehicle.setSpeed(veh_id, slow_speed)
                        elif chosen_speed < 0.9*slow_speed:
                            # print("Vehicle already slowed down: veh_id = ", veh_id, "dist = ", dist)
                            traci.vehicle.setSpeed(veh_id, -1)
                chosen_avs = new_chosen_avs




def output_file_to_df(output_file, num_reps=1):
    # Parse the XML file into pd dataframe
    tree = ET.parse(output_file)
    root = tree.getroot()

    dict = {"duration": [], "departDelay": [], "routeLength": [], "vType": [], "timeLoss": [], "id": []}
    for tripinfo in root.findall('tripinfo'):
        if tripinfo.get("id") == "stopping":
            continue
        for key in dict.keys():
            dict[key].append(tripinfo.get(key))
    df = pd.DataFrame(dict)
    df["speed"] = df.routeLength.astype(float) / df.duration.astype(float)
    df["totalDelay"] = df.departDelay.astype(float) + df.timeLoss.astype(float)
    df.drop(columns=["routeLength"], inplace=True)
    # convert to float except vType
    for col in df.columns:
        if col != "vType" and col != "id":
            df[col] = df[col].astype(float)

    return df

def calc_stats(df, diff=False):
    # Calculate statistics per vType
    metrics_stats = metrics
    if diff:
        metrics_stats = [f"{metric}_diff" for metric in metrics]
    stats = {}
    for vType in df.vType.unique():
        df_vType = df[df.vType == vType]
        stats[vType] = {}
        for metric in metrics_stats:
            stats[vType][f"avg_{metric}"] = df_vType[metric].mean()
            stats[vType][f"std_{metric}"] = df_vType[metric].std(ddof=1)
        stats[vType]["count"] = len(df_vType)
    if "all" not in stats.keys():
        stats["all"] = {}
        for metric in metrics_stats:
            stats["all"][f"avg_{metric}"] = df[metric].mean()
            stats["all"][f"std_{metric}"] = df[metric].std(ddof=1)
        stats["all"]["count"] = len(df)
    return pd.DataFrame(stats)


def parse_output_files_pairwise(args):
    av_rates, flow, policy_name1, policy_name2 = args
    # set MultiIndex for df - each vType will be a column in df with all the stats
    stats_names = [f"avg_{metric}_diff" for metric in metrics] + [f"std_{metric}_diff" for metric in metrics] + ["count"]
    vType_names = ["AV", "LaneChanger", "all"]
    df = pd.DataFrame(columns=pd.MultiIndex.from_product([vType_names, stats_names], names=['vType', 'stat']),
                      index=av_rates)

    for av_rate in av_rates:
        df_av_rate = pd.DataFrame()
        output_file1 = f"results_reps/{policy_name1}_{exp_name}_flow{flow}_av{av_rate}.xml"
        output_file2 = f"results_reps/{policy_name2}_{exp_name}_flow{flow}_av{av_rate}.xml"
        df_rep1 = output_file_to_df(output_file1)
        df_rep2 = output_file_to_df(output_file2)
        df_rep = pd.merge(df_rep1, df_rep2, on=["id","vType"], suffixes=[f"_{policy_name1}", f"_{policy_name2}"],
                          how="inner")
        # calculate difference
        try:
            assert len (df_rep) == len(df_rep1) == len(df_rep2)
        except:
            print(f"len(df_rep) = {len(df_rep)}, len(df_rep1) = {len(df_rep1)}, len(df_rep2) = {len(df_rep2)}")
            # print the ids that are not in both dataframes and the vTypes
            print(df_rep1[~df_rep1.id.isin(df_rep.id)][["id","vType"]])
            print("*"*50)
            print(df_rep2[~df_rep2.id.isin(df_rep.id)][["id","vType"]])
            print("*"*50)

        for metric in metrics:
            df_rep[f"{metric}_diff"] = ((df_rep[f"{metric}_{policy_name1}"] - df_rep[f"{metric}_{policy_name2}"])/\
                                       df_rep[f"{metric}_{policy_name2}"]) * 100
        df_rep.drop(columns=[f"{metric}_{policy_name1}" for metric in metrics], inplace=True)
        df_rep.drop(columns=[f"{metric}_{policy_name2}" for metric in metrics], inplace=True)
        df_av_rate = pd.concat([df_av_rate, df_rep])
        # Calculate statistics per vType
        stats_av_rate = calc_stats(df_av_rate, diff=True)
        # Add to df
        for vType in vType_names:
            if vType not in stats_av_rate.columns:
                continue
            for stat in stats_names:
                df.loc[av_rate, (vType, stat)] = stats_av_rate.loc[stat, vType]
    # Save df to csv
    df.to_csv(f"results_csvs/{policy_name1}_{policy_name2}_flow_{flow}.csv")
    df.to_pickle(f"results_csvs/{policy_name1}_{policy_name2}_flow_{flow}.pkl")

def parse_all_pairwise(policies, policy_name2,flows,av_rates):
    # run with pool for all flows and policies
    args = [(av_rates,flow, policy_name1,policy_name2) for flow in flows for policy_name1 in policies]
    with Pool(NUM_PROCESSES) as pool:
        results = list(tqdm(pool.imap(
            parse_output_files_pairwise, args), total=len(args)))
def convert_flows_to_av_rates(args):
    policy_name1, policy_name2, flows, av_rates = args
    # convert flows to av rates
    for av_rate in av_rates:
        stats_names = [f"avg_{metric}_diff" for metric in metrics] + [f"std_{metric}_diff" for metric in
                                                                      metrics] + ["count"]
        vType_names = ["AV", "HD", "Bus", "all"]
        df = pd.DataFrame(columns=pd.MultiIndex.from_product([vType_names, stats_names], names=['vType', 'stat']),
                          index=flows)
        for flow in flows:
            df_flow = pd.read_pickle(f"{results_folder}/{policy_name1}_{policy_name2}_flow_{flow}.pkl")
            df.loc[flow] = df_flow.loc[av_rate]
        df.to_csv(f"{results_folder}/{policy_name1}_{policy_name2}_av_rate_{av_rate}.csv")
        df.to_pickle(f"{results_folder}/{policy_name1}_{policy_name2}_av_rate_{av_rate}.pkl")

def convert_all_flows_to_av_rates(policies, policy_name2, flows, av_rates):
    args = [(policy_name1, policy_name2, flows, av_rates) for policy_name1 in policies]
    with Pool(NUM_PROCESSES) as pool:
        results = list(tqdm(pool.imap(
            convert_flows_to_av_rates, args), total=len(args)))

def create_results_table(args):
    # create the results table
    metric, vType, av_rates, flows, dist_slows, dist_fasts, slow_rates, stopping_lane = args
    if vType == "AV":
        av_rates.remove(0.0)
    if vType == "LaneChanger":
        av_rates.remove(1.0)
    cols = [f"flow_{flow}_av_rate_{av_rate}" for flow in flows for av_rate in av_rates]
    df = pd.DataFrame(columns=cols, index=[f"dist_slow_{dist_slow}_dist_fast_{dist_fast}_slow_rate_{slow_rate}" for dist_slow in dist_slows for dist_fast in dist_fasts for slow_rate in slow_rates])
    for flow in flows:
        for av_rate in av_rates:
            Nothing_df = output_file_to_df(
                f"{results_reps_folder}/Nothing_dist_slow_0_dist_fast_0_slow_rate_0_stopping_lane_{stopping_lane}_BlockedLane_flow{flow}_av{av_rate}.xml")
            for dist_slow in dist_slows:
                for dist_fast in dist_fasts:
                    for slow_rate in slow_rates:
                        relevant_df = output_file_to_df(f"{results_reps_folder}/SlowDown_dist_slow_{dist_slow}_dist_fast_{dist_fast}_slow_rate_{slow_rate}_stopping_lane_{stopping_lane}_BlockedLane_flow{flow}_av{av_rate}.xml")
                        # Merge the two dataframes
                        joined_df = pd.merge(relevant_df, Nothing_df, on=["id", "vType"], suffixes=["_SlowDown", "_Nothing"],
                                               how="inner")
                        for metric in metrics:
                            joined_df[f"{metric}_diff"] = ((joined_df[f"{metric}_SlowDown"] - joined_df[f"{metric}_Nothing"]) / \
                                                            joined_df[f"{metric}_Nothing"]) * 100
                        joined_df.drop(columns=[f"{metric}_SlowDown" for metric in metrics], inplace=True)
                        joined_df.drop(columns=[f"{metric}_Nothing" for metric in metrics], inplace=True)
                        if len(joined_df) != len(relevant_df):
                            print(f"len(relevant_df) = {len(relevant_df)}, len(Nothing_df) = {len(Nothing_df)}")
                            # print the ids that are not in both dataframes and the vTypes
                            print(relevant_df[~relevant_df.id.isin(Nothing_df.id)][["id", "vType"]])
                            print("*" * 50)
                            print(Nothing_df[~Nothing_df.id.isin(relevant_df.id)][["id", "vType"]])
                            print("*" * 50)
                        relevant_stats = calc_stats(joined_df, diff=True)
                        df.loc[f"dist_slow_{dist_slow}_dist_fast_{dist_fast}_slow_rate_{slow_rate}", f"flow_{flow}_av_rate_{av_rate}"] = relevant_stats.loc[f"avg_{metric}_diff", vType]
                        print(f"dist_slow_{dist_slow}_dist_fast_{dist_fast}_slow_rate_{slow_rate}, flow_{flow}_av_rate_{av_rate} = {relevant_stats.loc[f'avg_{metric}_diff', vType]}")
    df.to_csv(f"{results_folder}/{exp_name}_{metric}_{vType}_stopping_lane_{stopping_lane}.csv")

def create_all_results_tables(metrics, vTypes, av_rates, flows, dist_slows, dist_fasts, slow_rates, stopping_lane=1):
    # run over all metrics and vTypes with tqdm
    args = [(metric, vType, av_rates, flows, dist_slows, dist_fasts, slow_rates, stopping_lane) for metric in metrics for vType in vTypes]
    with Pool(NUM_PROCESSES) as pool:
        results = list(tqdm(pool.imap(
            create_results_table, args), total=len(args)))


if __name__ == '__main__':
    # define the parameters
    POLICIES = ["SlowDown", "Nothing"]
    DIST_SLOW_RANGE = [200, 300, 400, 500, 600, 700, 800]
    DIST_FAST_RANGE = [70]
    SLOW_RATE_RANGE = [0.6, 0.8]
    STOPPING_LANES = [1]
    AV_RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    FLOWS = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    # create the results tables
    create_all_results_tables(metrics, ["all","AV", "LaneChanger"], AV_RATES, FLOWS, DIST_SLOW_RANGE, DIST_FAST_RANGE, SLOW_RATE_RANGE, 1)