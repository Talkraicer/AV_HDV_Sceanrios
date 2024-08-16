import os
import xml.etree.ElementTree as ET
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import exp_name

NUM_PROCESSES = 70
results_folder = "results_csvs"
results_reps_folder = "results_reps"
METRICS = ["duration", "departDelay", "speed", "timeLoss", "totalDelay"]
VTYPES = ["AV", "HD", "Bus", "all"]

def output_file_to_df(output_file, num_reps=1):
    # Parse the XML file into pd dataframe
    tree = ET.parse(output_file)
    root = tree.getroot()

    dict = {"duration": [], "departDelay": [], "routeLength": [], "vType": [], "timeLoss": [], "id": [], "depart":[]}
    for tripinfo in root.findall('tripinfo'):
        for key in dict.keys():
            dict[key].append(tripinfo.get(key))
    df = pd.DataFrame(dict)
    df["speed"] = df.routeLength.astype(float) / df.duration.astype(float)
    df["totalDelay"] = df.departDelay.astype(float) + df.timeLoss.astype(float)
    df["vType"] = df["vType"].apply(lambda x: x.split("@")[0])
    df["numPass"] = df["vType"].apply(lambda x: x.split("_")[1])
    df["vType"] = df["vType"].apply(lambda x: x.split("_")[0])
    df["arrivalTime"] = df.duration.astype(float) + df.depart.astype(float)
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
    if "Passenger" not in stats.keys():
        stats["Passenger"] = {}
        # multiply each metric by the number of passengers
        for metric in metrics_stats:
            stats["Passenger"][f"avg_{metric}"] = df.apply(lambda x: x[metric] * x["numPass"], axis=1).mean()
            stats["Passenger"][f"std_{metric}"] = df.apply(lambda x: x[metric] * x["numPass"], axis=1).std(ddof=1)
        stats["Passenger"]["count"] = df["numPass"].sum()
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
    if "Passenger" not in stats.keys():
        stats["Passenger"] = {}
        # multiply each metric by the number of passengers
        stats["Passenger"][f"avg_{metric}"] = df.apply(lambda x: x[metric] * x["numPass"], axis=1).median()
        stats["Passenger"][f"std_{metric}"] = df.apply(lambda x: x[metric] * x["numPass"], axis=1).std(ddof=1)
        stats["Passenger"]["count"] = df["numPass"].sum()
    return pd.DataFrame(stats)


def create_results_table(args):
    # create the results table
    metric, vType, av_rate, policy_name = args
    policy_pure_name = policy_name.split("_")[0]
    if (vType.startswith("AV") and av_rate == 0.0) or (vType.startswith("HD") and av_rate == 1.0):
        return policy_name, av_rate, 0
    Nothing_df = output_file_to_df(f"{results_reps_folder}/Nothing{exp_name}_av{av_rate}.xml")
    relevant_df = output_file_to_df(f"{results_reps_folder}/{policy_name}{exp_name}_av{av_rate}.xml")
    # Merge the two dataframes
    joined_df = pd.merge(relevant_df, Nothing_df, on=["id", "vType"], suffixes=[f"_{policy_pure_name}", "_Nothing"],
                         how="inner")
    joined_df[f"{metric}_diff"] = ((joined_df[f"{metric}_{policy_pure_name}"] - joined_df[f"{metric}_Nothing"]) /
                                   joined_df[f"{metric}_Nothing"]) * 100
    assert len(joined_df) == len(relevant_df)
    relevant_stats = calc_stats_metric(joined_df, metric, diff=True)
    return policy_name, av_rate, relevant_stats.loc[f"avg_{metric}_diff", vType]


def create_all_results_tables(av_rates, policy_names):
    # run over all metrics and vTypes with tqdm
    for metric in tqdm(METRICS):
        for vType in tqdm(VTYPES, leave=False):
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
                df.loc[row_index, col_index] = value
            policy_pure_name = row_index.split("_")[0]
            os.makedirs(f"{results_folder}/{policy_pure_name}", exist_ok=True)
            df.to_csv(f"{results_folder}/{policy_pure_name}/{exp_name}_{policy_pure_name}_{metric}_{vType}.csv")


def parse_output_files(args):
    av_rates, num_reps, policy_name = args
    # Aggregate all output files into one dataframe, divided by vType

    # set MultiIndex for df - each vType will be a column in df with all the stats
    stats_names = [f"avg_{metric}" for metric in METRICS] + [f"std_{metric}" for metric in METRICS] + ["count"]
    vType_names = ["AV", "HD", "Bus", "all", "Passenger"]
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
    av_rates1, av_rate2, policy_name1, policy_baseline = args
    if av_rate2 in av_rates1:
        av_rates1.remove(av_rate2)
    # set MultiIndex for df - each vType will be a column in df with all the stats
    stats_names = [f"avg_{metric}_diff" for metric in METRICS] + [f"std_{metric}_diff" for metric in METRICS] + [
        "count"]
    vType_names = ["AV", "HD", "Bus", "all", "Passenger"]
    df = pd.DataFrame(columns=pd.MultiIndex.from_product([vType_names, stats_names], names=['vType', 'stat']),
                      index=av_rates1)

    for av_rate in av_rates1:
        df_av_rate = pd.DataFrame()
        output_file1 = f"results_reps/{policy_name1}{exp_name}_av{av_rate}.xml"
        av_rate2_dyn = av_rate2 if av_rate2 else av_rate
        output_file2 = f"results_reps/{policy_baseline}{exp_name}_av{av_rate2_dyn}.xml"
        df_rep1 = output_file_to_df(output_file1)
        df_rep2 = output_file_to_df(output_file2)
        df_rep = pd.merge(df_rep1, df_rep2, on=["id"], suffixes=[f"_{policy_name1}{av_rate}", f"_{policy_baseline}{av_rate2_dyn}"], how="inner")
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
            df_rep[f"{metric}_diff"] = ((df_rep[f"{metric}_{policy_name1}{av_rate}"] - df_rep[f"{metric}_{policy_baseline}{av_rate2_dyn}"]) /
                                        df_rep[f"{metric}_{policy_baseline}{av_rate2_dyn}"]) * 100
        df_rep.drop(columns=[f"{metric}_{policy_name1}{av_rate}" for metric in METRICS], inplace=True)
        df_rep.drop(columns=[f"{metric}_{policy_baseline}{av_rate2_dyn}" for metric in METRICS], inplace=True)

        df_rep["vType"] = df_rep[f"vType_{policy_name1}{av_rate}"]
        df_rep["numPass"] = df_rep[f"numPass_{policy_name1}{av_rate}"]

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
    if av_rate2:
        df.to_csv(f"results_csvs/{exp_name}_{policy_name1}_baseline_{policy_baseline}{av_rate2}.csv")
        df.to_pickle(f"results_csvs/{exp_name}_{policy_name1}_baseline_{policy_baseline}{av_rate2}.pkl")
    else:
        df.to_csv(f"results_csvs/{exp_name}_{policy_name1}_baseline_{policy_baseline}.csv")
        df.to_pickle(f"results_csvs/{exp_name}_{policy_name1}_baseline_{policy_baseline}.pkl")


def parse_all_output_files(av_rates, num_reps, policies):
    # run with pool for all flows and policies
    args = [(av_rates, num_reps, policy_name) for policy_name in policies]
    with Pool(NUM_PROCESSES) as pool:
        results = list(tqdm(pool.imap(
            parse_output_files, args), total=len(args)))


def parse_all_pairwise(policies, av_rates, policy_baseline="Nothing"):
    # run with pool for all flows and policies
    args = [(av_rates, None, policy_name1, policy_baseline) for policy_name1 in policies]
    # args += [(av_rates, 1.0, policy_name1, policy_baseline) for policy_name1 in policies]
    # args += [(av_rates, 1.0, policy_name1) for policy_name1 in policies]
    with Pool(NUM_PROCESSES) as pool:
        results = list(tqdm(pool.imap(
            parse_output_files_pairwise, args), total=len(args)))


def convert_flows_to_av_rates(args):
    policy_name1, policy_name2, flows, av_rates = args
    # convert flows to av rates
    for av_rate in av_rates:
        stats_names = [f"avg_{metric}_diff" for metric in METRICS] + [f"std_{metric}_diff" for metric in
                                                                      METRICS] + ["count"]
        vType_names = ["AV", "HD", "Bus", "all", "Passenger"]
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


SCENARIO_START_TIMES = [0,10800,16200,23400,30600, np.inf]
SCENARIO_NAMES = ["RUSH_HOUR_EXT","PEAK","MID_DAY","WEEKEND", "MODERATE_RUSH_HOUR"]
def parse_scenarios(output_files):
    policy_names = []
    av_rates = []
    for output_file in output_files:
        file_name = output_file.split("/")[-1]
        policy_name = file_name[:file_name.find(exp_name)]
        av_rate = file_name[file_name.find("av"):file_name.find(".xml")]
        policy_names.append(policy_name)
        av_rates.append(av_rate)
    av_rates = list(set(av_rates))

    df = pd.DataFrame(columns=pd.MultiIndex.from_product([SCENARIO_NAMES, av_rates]),
                      index=policy_names)
    for output_file in tqdm(output_files):
        file_name = output_file.split("/")[-1]
        policy_name = file_name[:file_name.find(exp_name)]
        av_rate = file_name[file_name.find("av"):file_name.find(".xml")]
        df = output_file_to_df(output_file)
        for scenario in SCENARIO_NAMES:
            start_time = SCENARIO_START_TIMES[SCENARIO_NAMES.index(scenario)]
            end_time = SCENARIO_START_TIMES[SCENARIO_NAMES.index(scenario)+1]
            df_scenario = df[(df.depart >= start_time) & (df.depart < end_time)]
            total_delay_timestamp = calc_stats_metric(df_scenario, "totalDelay", diff=False)
            mean_pass_delay_scenario = total_delay_timestamp.loc["avg_totalDelay", "Passenger"]
            df.loc[policy_name, (scenario, av_rate)] = mean_pass_delay_scenario
    df.to_csv(f"{results_folder}/scenarios_{exp_name}.csv")

if __name__ == '__main__':
    # Example usage
    # AV_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    #
    # policies = ["Nothing"]+[f"StaticNumPass_{i}" for i in range(1, 6)]
    # # parse_all_output_files(AV_rates, 1, policies)
    # policies.remove("Nothing")
    # parse_all_pairwise(policies, AV_rates)
    # policies.remove("Nothing")
    # create_all_results_tables(AV_rates, policies)
    # STOP_FROM_RANGE = [300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
    # STOP_TO_RANGE = [0, 100, 200]
    # policies = ["DisallowBack"]
    # policy_names = [f"{policy}_{stop_from}_{stop_to}" for policy in policies for stop_from in STOP_FROM_RANGE for stop_to in STOP_TO_RANGE]
    # create_all_results_tables(AV_rates, policy_names)
    parse_scenarios(["results_reps/"+f for f in os.listdir("results_reps") if f.find(exp_name) != -1])