import os
import traci
import imageio
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

exp_name = "PublicTransport"
NUM_PROCESSES = 70
GUI = True
sumoCfg = fr"../{exp_name}.sumocfg"
results_folder = "results_csvs"
metrics = ["duration", "departDelay", "speed", "timeLoss", "totalDelay"]

def clear_front_of_vehicle(vehID, lane, limit = np.inf):
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

def handle_step(t, policy_name):
    vehIDs = traci.vehicle.getIDList()
    has_emergency = False
    for vehID in vehIDs:
        lane = traci.vehicle.getLaneID(vehID)
        if traci.vehicle.getTypeID(vehID) == "Bus":
            if policy_name == "ClearFront500":
                clear_front_of_vehicle(vehID, lane, 500)
            elif policy_name == "ClearFront":
                clear_front_of_vehicle(vehID, lane)
            elif policy_name == "ClearFront100":
                clear_front_of_vehicle(vehID, lane, 100)
    return has_emergency


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


def parse_output_files(av_rates, num_reps, policy_name, flow):
    # Aggregate all output files into one dataframe, divided by vType

    # set MultiIndex for df - each vType will be a column in df with all the stats
    stats_names = [f"avg_{metric}" for metric in metrics] + [f"std_{metric}" for metric in metrics]+ ["count"]
    vType_names = ["AV", "HD", "Bus", "all"]
    df = pd.DataFrame(columns=pd.MultiIndex.from_product([vType_names, stats_names], names=['vType', 'stat']),
                      index=av_rates)

    for av_rate in av_rates:
        df_av_rate = pd.DataFrame()
        for i in range(num_reps):
            output_file = f"results_reps/{policy_name}_flow_{flow}_av_rate_{av_rate}_rep_{i}.xml"
            if num_reps == 1:
                output_file = f"results_reps_long/{policy_name}_flow_{flow}_av_rate_{av_rate}_rep_{i}.xml"
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
    if num_reps == 1:
        df.to_csv(f"results_csvs/{policy_name}_flow_{flow}_long.csv")
        df.to_pickle(f"results_csvs/{policy_name}_flow_{flow}_long.pkl")
        return
    df.to_csv(f"results_csvs/{policy_name}_flow_{flow}.csv")
    df.to_pickle(f"results_csvs/{policy_name}_flow_{flow}.pkl")


def parse_output_files_pairwise(args, bus_prob=0.1):
    av_rates, flow, policy_name1, policy_name2 = args
    # set MultiIndex for df - each vType will be a column in df with all the stats
    stats_names = [f"avg_{metric}_diff" for metric in metrics] + [f"std_{metric}_diff" for metric in metrics] + ["count"]
    vType_names = ["AV", "HD", "Bus", "all"]
    df = pd.DataFrame(columns=pd.MultiIndex.from_product([vType_names, stats_names], names=['vType', 'stat']),
                      index=av_rates)

    for av_rate in av_rates:
        df_av_rate = pd.DataFrame()
        output_file1 = f"results_reps/{policy_name1}_PublicTransport_flow{flow}_av{av_rate}_Bus{bus_prob}.xml"
        output_file2 = f"results_reps/{policy_name2}_PublicTransport_flow{flow}_av{av_rate}_Bus{bus_prob}.xml"
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
    df.to_parquet(f"results_csvs/{policy_name1}_{policy_name2}_flow_{flow}.pt")

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
        df.to_parquet(f"{results_folder}/{policy_name1}_{policy_name2}_av_rate_{av_rate}.pt")

def convert_all_flows_to_av_rates(policies, policy_name2, flows, av_rates):
    args = [(policy_name1, policy_name2, flows, av_rates) for policy_name1 in policies]
    with Pool(NUM_PROCESSES) as pool:
        results = list(tqdm(pool.imap(
            convert_flows_to_av_rates, args), total=len(args)))



if __name__ == '__main__':
    # Example usage
    AV_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    Bus_prob = 0.1
    FLOWS = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    policies = ["ClearFront", "ClearFront500", "ClearFront100"]
    parse_all_pairwise(policies, "Nothing", FLOWS, AV_rates)
    parse_all_pairwise(policies, "NothingDL", FLOWS, AV_rates)
    convert_all_flows_to_av_rates(policies, "Nothing", FLOWS, AV_rates)
    convert_all_flows_to_av_rates(policies, "NothingDL", FLOWS, AV_rates)
