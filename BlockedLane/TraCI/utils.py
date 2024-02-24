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
METRICS = ["duration", "departDelay", "speed", "timeLoss", "totalDelay"]

chosen_avs = {}


def is_leader_AV_before_stopper(veh_id):
    # check if the vehicle is the leader of the AVs before the stopper
    if traci.vehicle.getTypeID(veh_id) == "AV":
        dist = 0
        leader = traci.vehicle.getLeader(veh_id, 0)
        while leader is not None:
            frontVehID, dist_leader = leader
            dist += dist_leader
            if frontVehID == "stopping" and traci.vehicle.getLaneID(veh_id) == traci.vehicle.getLaneID(frontVehID):
                return True, dist
            leader = traci.vehicle.getLeader(frontVehID, 0)
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
                arg_min_dist_to_fast = veh_id, dist
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
                        continue

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
                        elif chosen_speed < 0.9 * slow_speed:
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


def calc_stats(df, metric, diff=False):
    # Calculate statistics per vType
    if diff:
        metric = f"{metric}_diff"
    stats = {}
    for vType in df.vType.unique():
        df_vType = df[df.vType == vType]
        stats[vType] = {}
        stats[vType][f"avg_{metric}"] = df_vType[metric].median()
        stats[vType][f"std_{metric}"] = df_vType[metric].std(ddof=1)
        stats[vType]["count"] = len(df_vType)
    if "all" not in stats.keys():
        stats["all"] = {}
        stats["all"][f"avg_{metric}"] = df[metric].median()
        stats["all"][f"std_{metric}"] = df[metric].std(ddof=1)
        stats["all"]["count"] = len(df)
    return pd.DataFrame(stats)


def create_results_table(args):
    # create the results table
    metric, vType, av_rate, flow, dist_slow, dist_fast, slow_rate, stopping_lane = args
    if (vType == "AV" and av_rate == 0.0) or (vType == "LaneChanger" and av_rate == 1.0):
        return
    Nothing_df = output_file_to_df(f"{results_reps_folder}/Nothing_dist_slow_0_dist_fast_0_slow_rate_0_stopping_lane_"
                                   f"{stopping_lane}_BlockedLane_flow{flow}_av{av_rate}.xml")
    relevant_df = output_file_to_df(f"{results_reps_folder}/SlowDown_dist_slow_{dist_slow}_dist_fast_{dist_fast}"
                                    f"_slow_rate_{slow_rate}_stopping_lane_{stopping_lane}_BlockedLane_flow{flow}"
                                    f"_av{av_rate}.xml")
    # Merge the two dataframes
    joined_df = pd.merge(relevant_df, Nothing_df, on=["id", "vType"], suffixes=["_SlowDown", "_Nothing"], how="inner")
    joined_df[f"{metric}_diff"] = ((joined_df[f"{metric}_SlowDown"] - joined_df[f"{metric}_Nothing"]) /
                                   joined_df[f"{metric}_Nothing"]) * 100
    assert len(joined_df) == len(relevant_df)
    relevant_stats = calc_stats(joined_df, metric, diff=True)
    return (dist_slow,dist_fast,slow_rate),(flow,av_rate),relevant_stats.loc[f"avg_{metric}_diff", vType]


def create_all_results_tables(metrics, vTypes, av_rates, flows, dist_slows, dist_fasts, slow_rates, stopping_lane=1):
    # run over all metrics and vTypes with tqdm
    for metric in tqdm(metrics):
        for vType in tqdm(vTypes,leave=False):
            args = [(metric, vType, av_rate, flow, dist_slow, dist_fast, slow_rate, stopping_lane)
                    for av_rate in av_rates for flow in flows for dist_slow in dist_slows for dist_fast
                    in dist_fasts for slow_rate in slow_rates]
            with Pool(NUM_PROCESSES) as pool:
                results = list(tqdm(pool.imap(
                    create_results_table, args), total=len(args)))
            cols = [f"flow_{flow}_av_rate_{av_rate}" for flow in flows for av_rate in av_rates]
            df = pd.DataFrame(columns=cols,
                              index=[f"dist_slow_{dist_slow}_dist_fast_{dist_fast}_slow_rate_{slow_rate}" for dist_slow in
                                     dist_slows for dist_fast in dist_fasts for slow_rate in slow_rates])
            for result in results:
                row, col, value = result
                row_index = "dist_slow_{}_dist_fast_{}_slow_rate_{}".format(*row)
                col_index = "flow_{}_av_rate_{}".format(*col)
                df.loc[row_index,col_index] = value

            df.to_csv(f"results_csvs/BlockedLane_{metric}_{vType}_stopping_lane_{stopping_lane}.csv")


if __name__ == '__main__':
    # test
    # args = ["duration", "all", [0.7], [8000], [200], [70], [0.6], 1]
    # create_results_table(args)

    # define the parameters
    POLICIES = ["SlowDown", "Nothing"]
    DIST_SLOW_RANGE = [200, 300, 400, 500, 600, 700, 800]
    DIST_FAST_RANGE = [70, 100, 150, 200]
    SLOW_RATE_RANGE = [0.6, 0.8]
    STOPPING_LANES = [0]
    AV_RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    FLOWS = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    # create the results tables
    create_all_results_tables(METRICS, ["all", "AV", "LaneChanger"], AV_RATES, FLOWS, DIST_SLOW_RANGE, DIST_FAST_RANGE,
                              SLOW_RATE_RANGE, STOPPING_LANES[0])
