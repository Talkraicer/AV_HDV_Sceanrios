import os
import traci
import imageio
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

exp_name = "PublicTransportV2Bay"
NUM_PROCESSES = 70
GUI = False
sumoCfg = fr"../{exp_name}.sumocfg"
results_folder = "results_csvs"
results_reps_folder = "results_reps"
METRICS = ["duration", "departDelay", "speed", "timeLoss", "totalDelay"]
VTYPES = ["AV", "HD", "Bus", "all"]


def clear_front_of_vehicle(vehID, lane, limit=np.inf):
    leader = traci.vehicle.getLeader(vehID, 0)
    dist_emer = 0
    while leader and dist_emer < limit:
        frontVehID, dist = leader
        dist_emer += dist
        if traci.vehicle.getTypeID(frontVehID) == "AV" and traci.vehicle.getLaneID(frontVehID) == lane and \
                dist_emer < limit:
            to_lane = 1 if lane.endswith("2") else 0 if lane.endswith("1") else 1
            traci.vehicle.changeLane(frontVehID, to_lane, 1)
        leader = traci.vehicle.getLeader(frontVehID, 0)


def get_stopping_buses_ids():
    # get ids of buses that are stopping
    vehIDs = traci.vehicle.getIDList()
    stopping_buses = []
    for vehID in vehIDs:
        if traci.vehicle.getTypeID(vehID) == "Bus" and traci.vehicle.getSpeed(vehID) < 0.1:
            stopping_buses.append(vehID)
    return stopping_buses


def vehicles_distance(vehID1, vehID2):
    # get the distance between two vehicles, may be negative if vehID1 is behind vehID2
    pos1 = traci.vehicle.getPosition(vehID1)
    pos2 = traci.vehicle.getPosition(vehID2)
    return pos1[0] - pos2[0]


def check_disallow_back(vehID, stopping_buses, stop_from, stop_to):
    # check if the vehicle is behind a bus that is stopping
    if not stopping_buses:
        return False
    for busID in stopping_buses:
        if -stop_from <= vehicles_distance(vehID, busID) <= -stop_to:
            return True
    return False

def handle_step(t, policy_name):
    if policy_name.startswith("DisallowBack"):
        stop_from = int(policy_name.split("_")[1])
        stop_to = int(policy_name.split("_")[2])
        vehIDs = traci.vehicle.getIDList()
        stopping_buses = get_stopping_buses_ids()
        for vehID in vehIDs:
            if traci.vehicle.getTypeID(vehID).startswith("AV"):
                if check_disallow_back(vehID, stopping_buses, stop_from, stop_to):
                    # set type to temporalHD
                    traci.vehicle.setType(vehID, "TemporalHD")
                    traci.vehicle.setVehicleClass(vehID,"passenger")
            elif traci.vehicle.getTypeID(vehID).startswith("TemporalHD"):
                if not check_disallow_back(vehID, stopping_buses, stop_from, 0):
                    # set type to AV
                    traci.vehicle.setType(vehID, "AV")
                    traci.vehicle.setVehicleClass(vehID,"evehicle")

            if policy_name.startswith("DisallowBackRelease"):
                if traci.vehicle.getTypeID(vehID).startswith("TemporalHD"):
                    if check_disallow_back(vehID, stopping_buses, 30, 0) and \
                        (traci.vehicle.getLaneID(vehID).endswith("0") or
                         (traci.vehicle.getLaneID(vehID).find(".S") != -1 and traci.vehicle.getLaneID(vehID).endswith("1"))):
                        traci.vehicle.setType(vehID, "AV")
                        traci.vehicle.setVehicleClass(vehID, "evehicle")

def output_file_to_df(output_file, num_reps=1):
    # Parse the XML file into pd dataframe
    tree = ET.parse(output_file)
    root = tree.getroot()

    dict = {"duration": [], "departDelay": [], "routeLength": [], "vType": [], "timeLoss": [], "id": []}
    for tripinfo in root.findall('tripinfo'):
        for key in dict.keys():
            dict[key].append(tripinfo.get(key))
    df = pd.DataFrame(dict)
    df["speed"] = df.routeLength.astype(float) / df.duration.astype(float)
    df["totalDelay"] = df.departDelay.astype(float) + df.timeLoss.astype(float)
    df["vType"] = df["vType"].apply(lambda x: x.split("@")[0])
    df.drop(columns=["routeLength"], inplace=True)
    # convert to float except vType
    for col in df.columns:
        if col != "vType" and col != "id":
            df[col] = df[col].astype(float)
    if num_reps > 1:
        df = calc_mean(df)
    return df


def calc_mean(df):
    df_results = df.groupby(by="vType").mean().reset_index()
    # append vType column of all vehicles
    df_all = df.drop(columns=["vType"]).mean()
    df_all["vType"] = "all"
    df_all = pd.DataFrame(df_all).transpose()
    df_results = pd.concat([df_results, df_all]).reset_index(drop=True)
    return df_results


def calc_stats(df, diff=False):
    # Calculate statistics per vType
    metrics_stats = METRICS
    if diff:
        metrics_stats = [f"{metric}_diff" for metric in METRICS]
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
def calc_stats_metric(df, metric, diff=False):
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
    metric, vType, av_rate, policy_name = args
    policy_pure_name = policy_name.split("_")[0]
    if (vType == "AV" and av_rate == 0.0) or (vType == "HD" and av_rate == 1.0):
        return policy_name,av_rate, 0
    Nothing_df = output_file_to_df(f"{results_reps_folder}/Nothing_{exp_name}_av{av_rate}.xml")
    relevant_df = output_file_to_df(f"{results_reps_folder}/{policy_name}{exp_name}_av{av_rate}.xml")
    # Merge the two dataframes
    joined_df = pd.merge(relevant_df, Nothing_df, on=["id", "vType"], suffixes=[f"_{policy_pure_name}", "_Nothing"], how="inner")
    joined_df[f"{metric}_diff"] = ((joined_df[f"{metric}_{policy_pure_name}"] - joined_df[f"{metric}_Nothing"]) /
                                   joined_df[f"{metric}_Nothing"]) * 100
    assert len(joined_df) == len(relevant_df)
    relevant_stats = calc_stats_metric(joined_df, metric, diff=True)
    return policy_name,av_rate,relevant_stats.loc[f"avg_{metric}_diff", vType]


def create_all_results_tables(av_rates, policy_names):
    # run over all metrics and vTypes with tqdm
    for metric in tqdm(METRICS):
        for vType in tqdm(VTYPES,leave=False):
            args = [(metric, vType, av_rate, policy_name)
                    for av_rate in av_rates for policy_name in policy_names]
            with Pool(NUM_PROCESSES) as pool:
                results = list(tqdm(pool.imap(
                    create_results_table, args), total=len(args)))
            cols = [av_rate for av_rate in av_rates]
            df = pd.DataFrame(columns=cols,
                              index=[policy_name for policy_name in policy_names])
            for result in results:
                row_index, col_index, value = result
                df.loc[row_index,col_index] = value
            policy_pure_name = row_index.split("_")[0]
            os.makedirs(f"{results_folder}/{policy_pure_name}", exist_ok=True)
            df.to_csv(f"{results_folder}/{policy_pure_name}/{exp_name}_{policy_pure_name}_{metric}_{vType}.csv")


def parse_output_files(av_rates, num_reps, policy_name):
    # Aggregate all output files into one dataframe, divided by vType

    # set MultiIndex for df - each vType will be a column in df with all the stats
    stats_names = [f"avg_{metric}" for metric in METRICS] + [f"std_{metric}" for metric in METRICS] + ["count"]
    vType_names = ["AV", "HD", "Bus", "all"]
    df = pd.DataFrame(columns=pd.MultiIndex.from_product([vType_names, stats_names], names=['vType', 'stat']),
                      index=av_rates)

    for av_rate in av_rates:
        df_av_rate = pd.DataFrame()
        output_file = f"results_reps/{policy_name}{exp_name}_av{av_rate}.xml"
        df_rep = output_file_to_df(output_file, num_reps)
        df_av_rate = pd.concat([df_av_rate, df_rep])
        # Calculate statistics per vType
        stats_av_rate = calc_stats(df_av_rate)
        # Add to df
        for vType in vType_names:
            if vType not in stats_av_rate.columns:
                continue
            for stat in stats_names:
                df.loc[av_rate, (vType, stat)] = stats_av_rate.loc[stat, vType]
    # Save df to csv
    df.to_csv(f"results_csvs/{policy_name}_{exp_name}.csv")
    df.to_pickle(f"results_csvs/{policy_name}_{exp_name}.pkl")


def parse_output_files_pairwise(args):
    av_rates1, av_rate2, policy_name1 = args
    av_rates1.remove(av_rate2)
    # set MultiIndex for df - each vType will be a column in df with all the stats
    stats_names = [f"avg_{metric}_diff" for metric in METRICS] + [f"std_{metric}_diff" for metric in METRICS] + [
        "count"]
    vType_names = ["AV", "HD", "Bus", "all"]
    df = pd.DataFrame(columns=pd.MultiIndex.from_product([vType_names, stats_names], names=['vType', 'stat']),
                      index=av_rates1)

    for av_rate in av_rates1:
        df_av_rate = pd.DataFrame()
        output_file1 = f"results_reps/{policy_name1}{exp_name}_av{av_rate}.xml"
        output_file2 = f"results_reps/{policy_name1}{exp_name}_av{av_rate2}.xml"
        df_rep1 = output_file_to_df(output_file1)
        df_rep2 = output_file_to_df(output_file2)
        df_rep = pd.merge(df_rep1, df_rep2, on=["id"], suffixes=[f"_{av_rate}", f"_{av_rate2}"],
                          how="inner")
        # calculate difference
        try:
            assert len(df_rep) == len(df_rep1) == len(df_rep2)
        except:
            print(f"len(df_rep) = {len(df_rep)}, len(df_rep1) = {len(df_rep1)}, len(df_rep2) = {len(df_rep2)}")
            # print the ids that are not in both dataframes and the vTypes
            print(df_rep1[~df_rep1.id.isin(df_rep.id)][["id", "vType"]])
            print("*" * 50)
            print(df_rep2[~df_rep2.id.isin(df_rep.id)][["id", "vType"]])
            print("*" * 50)

        for metric in METRICS:
            df_rep[f"{metric}_diff"] = ((df_rep[f"{metric}_{av_rate}"] - df_rep[f"{metric}_{av_rate2}"]) /
                                        df_rep[f"{metric}_{av_rate2}"]) * 100
        df_rep.drop(columns=[f"{metric}_{av_rate}" for metric in METRICS], inplace=True)
        df_rep.drop(columns=[f"{metric}_{av_rate2}" for metric in METRICS], inplace=True)

        df_rep["vType"] = df_rep[f"vType_{av_rate}"]
        df_rep.drop(columns=[f"vType_{av_rate}", f"vType_{av_rate2}"], inplace=True)

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
    df.to_csv(f"results_csvs/{exp_name}_{policy_name1}_baseline{av_rate2}.csv")
    df.to_pickle(f"results_csvs/{exp_name}_{policy_name1}_baseline{av_rate2}.pkl")


def parse_all_pairwise(policies, av_rates):
    # run with pool for all flows and policies
    args = [(av_rates, 0.0, policy_name1) for policy_name1 in policies]
    args += [(av_rates, 1.0, policy_name1) for policy_name1 in policies]
    with Pool(NUM_PROCESSES) as pool:
        results = list(tqdm(pool.imap(
            parse_output_files_pairwise, args), total=len(args)))


def convert_flows_to_av_rates(args):
    policy_name1, policy_name2, flows, av_rates = args
    # convert flows to av rates
    for av_rate in av_rates:
        stats_names = [f"avg_{metric}_diff" for metric in METRICS] + [f"std_{metric}_diff" for metric in
                                                                      METRICS] + ["count"]
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


if __name__ == '__main__':
    # Example usage
    AV_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    policies = ["DisallowBackRelease30_1000_200"]
    parse_output_files(AV_rates, 1, policies[0])
    parse_all_pairwise(policies, AV_rates)
    # STOP_FROM_RANGE = [300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
    # STOP_TO_RANGE = [0, 100, 200]
    # policies = ["DisallowBack"]
    # policy_names = [f"{policy}_{stop_from}_{stop_to}" for policy in policies for stop_from in STOP_FROM_RANGE for stop_to in STOP_TO_RANGE]
    # create_all_results_tables(AV_rates, policy_names)