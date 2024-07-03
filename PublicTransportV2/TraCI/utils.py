import os
import traci
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import wandb

exp_name = "LeftComp"
NUM_PROCESSES = 70
GUI = False
sumoCfg = fr"../{exp_name}.sumocfg"
results_folder = "results_csvs"
results_reps_folder = "results_reps"
METRICS = ["duration", "departDelay", "speed", "timeLoss", "totalDelay"]
VTYPES = ["AV", "HD", "Bus", "all"]
MAX_VEH_SPEED = 55.56

MERGING_TIME_FACTOR = 1.2
HOLDING_TIME_FACTOR = 1.2
BUS_STOPPING_TIME = 25
MAX_ALLOWED_SPEED = 25
STOP_FROM = 1000

BUSES_VOLUNTEERS = dict()

# visualization effects
PT_LANE_SPEED_GRAPH = True
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
        if traci.vehicle.getTypeID(vehID).find("Bus") != -1 and traci.vehicle.getSpeed(
                vehID) == 0 and traci.vehicle.getLaneID(vehID).endswith("S_0"):
            stopping_buses.append(vehID)
    return stopping_buses


def vehicles_distance(vehID1, vehID2):
    # get the distance between two vehicles, may be negative if vehID1 is behind vehID2
    try:
        pos1 = traci.vehicle.getPosition(vehID1)
        pos2 = traci.vehicle.getPosition(vehID2)
        return pos1[0] - pos2[0]
    except:
        with open("errors.txt", "a+") as f:
            f.write(f"vehID1 = {vehID1}, vehID2 = {vehID2} had error\n")
        return np.inf


def check_disallow_back(vehID, stopping_buses, stop_from, stop_to):
    # check if the vehicle is behind a bus that is stopping
    if not stopping_buses:
        return False
    for busID in stopping_buses:
        if -stop_from <= vehicles_distance(vehID, busID) <= -stop_to:
            return True
    return False


def switch_to_temporalHD(vehID):
    traci.vehicle.setType(vehID, "TemporalHD")


def switch_to_allowedTemporalHD(vehID):
    traci.vehicle.setType(vehID, "AllowedTemporalHD")


def switch_to_AV(vehID):
    traci.vehicle.setType(vehID, "AV")
    traci.vehicle.setVehicleClass(vehID, "evehicle")


def assign_volunteer(busID):
    global BUSES_VOLUNTEERS
    vehIDs = traci.vehicle.getIDList()
    max_estimated_time = 0
    volunteer = None
    for vehID in vehIDs:
        if traci.vehicle.getTypeID(vehID).startswith("AV") and traci.vehicle.getLaneID(vehID).endswith("0"):
            if traci.vehicle.getSpeed(vehID) == 0:
                continue
            estimated_time_to_reach = vehicles_distance(busID, vehID) / traci.vehicle.getSpeed(vehID)
            if 0.5 * BUS_STOPPING_TIME < estimated_time_to_reach < BUS_STOPPING_TIME * 1.2:
                if estimated_time_to_reach > max_estimated_time:
                    max_estimated_time = estimated_time_to_reach
                    volunteer = vehID
    BUSES_VOLUNTEERS[busID] = volunteer
    if volunteer:
        traci.vehicle.setColor(volunteer, (0, 255, 0))
        traci.vehicle.changeLane(volunteer, 0, BUS_STOPPING_TIME * HOLDING_TIME_FACTOR)
        traci.vehicle.setMaxSpeed(volunteer,
                                  vehicles_distance(busID, volunteer) / (BUS_STOPPING_TIME * MERGING_TIME_FACTOR))


def release_volunteer(volID):
    if volID:
        try:
            traci.vehicle.setColor(volID, (0, 0, 255))
            traci.vehicle.setMaxSpeed(volID, MAX_VEH_SPEED)
            traci.vehicle.changeLane(volID, 0, 0)
        except:
            with open("erros.txt", "a+") as f:
                f.write(f"volID = {volID} had error\n")


def count_avs_buses(dist):
    # count the number of AVs and buses in the range (0,dist). Count non-stopping buses as AVs
    vehIDs = traci.vehicle.getIDList()
    num_AVs = 0
    num_buses = 0
    for vehID in vehIDs:
        pos = traci.vehicle.getPosition(vehID)
        if 0 < pos[0] < dist:
            if traci.vehicle.getTypeID(vehID).startswith("AV") or vehID.startswith("bus_nonstop"):
                num_AVs += 1
            elif vehID.startswith("bus_stop"):
                num_buses += 1
    return num_AVs, num_buses


def handle_step(t, policy_name,av_rate):
    global BUSES_VOLUNTEERS
    if policy_name == "Nothing" and exp_name.startswith("Left") and t < 1:
        for lane in traci.lane.getIDList():
            if "bus" in traci.lane.getAllowed(lane):
                traci.lane.setAllowed(lane, "bus")

    if policy_name.startswith("DisallowBack"):
        stop_from = int(policy_name.split("_")[1])
        stop_to = int(policy_name.split("_")[2])
        vehIDs = traci.vehicle.getIDList()
        stopping_buses = get_stopping_buses_ids()
        for vehID in vehIDs:
            laneID = traci.vehicle.getLaneID(vehID)
            typeID = traci.vehicle.getTypeID(vehID)
            if typeID.startswith("AV"):
                if check_disallow_back(vehID, stopping_buses, stop_from, stop_to):
                    if laneID.endswith("0") and laneID.find(".S") == -1:
                        switch_to_allowedTemporalHD(vehID)
                    elif not laneID.endswith("0") and not laneID.endswith("S_1"):
                        switch_to_temporalHD(vehID)
            elif typeID.find("TemporalHD") != -1:
                if not check_disallow_back(vehID, stopping_buses, stop_from, 0) \
                        or laneID.endswith("S_1"):
                    switch_to_AV(vehID)
                elif typeID.find("AllowedTemporalHD") != -1 and not laneID.endswith("0") and not laneID.endswith("S_1"):
                    switch_to_temporalHD(vehID)

    if policy_name == "Volunteer_Stopper":
        vehIDs = traci.vehicle.getIDList()
        stopping_buses = get_stopping_buses_ids()

        # assign volunteers to new stopping buses
        for bus in stopping_buses:
            if bus not in BUSES_VOLUNTEERS.keys():
                assign_volunteer(bus)

        # release volunteers if the bus is not stopping anymore
        to_del = []
        for bus in BUSES_VOLUNTEERS.keys():
            if bus not in stopping_buses:
                release_volunteer(BUSES_VOLUNTEERS[bus])
                to_del.append(bus)
        for bus_del in to_del:
            BUSES_VOLUNTEERS.pop(bus_del)

        # check if the AVs need to switch to TemporalHD
        for vehID in vehIDs:
            laneID = traci.vehicle.getLaneID(vehID)
            typeID = traci.vehicle.getTypeID(vehID)
            was_THD = False
            if typeID.find("TemporalHD") != -1:
                switch_to_AV(vehID)
                was_THD = True
            if typeID.startswith("AV") or was_THD:
                for stopped_bus in BUSES_VOLUNTEERS:
                    if BUSES_VOLUNTEERS[stopped_bus]:
                        if vehicles_distance(vehID, BUSES_VOLUNTEERS[stopped_bus]) > 0 and \
                                vehicles_distance(vehID, stopped_bus) < 0 and \
                                not laneID.endswith("0") and \
                                not laneID.find(".S") != -1:
                            switch_to_temporalHD(vehID)
                            break
                    elif (0 < vehicles_distance(stopped_bus, vehID) < BUS_STOPPING_TIME * MAX_ALLOWED_SPEED and
                          not laneID.endswith("0") and not laneID.find(".S") != -1
                          and vehID not in BUSES_VOLUNTEERS.values()):
                        switch_to_temporalHD(vehID)
                        break
    if policy_name.startswith("FastLane"):
        dist_features = int(policy_name.split("_")[1])
        num_avs_max = int(policy_name.split("_")[2])
        num_buses_max = int(policy_name.split("_")[3])
        num_avs, num_buses = count_avs_buses(dist_features)
        insert_vehicles = num_avs <= num_avs_max and num_buses <= num_buses_max

        for vehID in traci.vehicle.getIDList():
            pos = traci.vehicle.getPosition(vehID)[0]
            vType = traci.vehicle.getTypeID(vehID)
            laneID = traci.vehicle.getLaneID(vehID)
            if vType.startswith("AV") and pos < 7800:
                if 100 < pos and not laneID.endswith("0"):
                    switch_to_temporalHD(vehID)
                else:
                    if insert_vehicles:
                        if not laneID.endswith("0"):
                            traci.vehicle.changeLane(vehID, 0, 1)
                        elif pos > 0:
                            switch_to_allowedTemporalHD(vehID)
                    else:
                        switch_to_temporalHD(vehID)
            if vType.find("TemporalHD") != -1 and pos > 7800:
                switch_to_AV(vehID)

    if policy_name.startswith("EnterClear"):
        dist = int(policy_name.split("_")[1])
        vehIDs = traci.vehicle.getIDList()
        for vehID in vehIDs:
            laneID = traci.vehicle.getLaneID(vehID)
            typeID = traci.vehicle.getTypeID(vehID)
            pos = traci.vehicle.getPosition(vehID)[0]
            if typeID.startswith("Bus") and pos < -450:
                for vehID2 in vehIDs:
                    typeID2 = traci.vehicle.getTypeID(vehID2)
                    pos2 = traci.vehicle.getPosition(vehID2)[0]
                    if typeID2.startswith("AV") and pos < pos2 < pos + dist:
                        switch_to_temporalHD(vehID2)
                break
        for vehID in vehIDs:
            typeID = traci.vehicle.getTypeID(vehID)
            if typeID.find("TemporalHD") != -1 and traci.vehicle.getPosition(vehID)[0] > 1500:
                switch_to_AV(vehID)
    if policy_name.startswith("StaticNumPass"):
        min_num_pass = int(policy_name.split("_")[1][0])
        vehIDs = traci.vehicle.getIDList()
        for vehID in vehIDs:
            typeID = traci.vehicle.getTypeID(vehID)
            if typeID.startswith("AV") and int(typeID.split("_")[1][0]) < min_num_pass:
                traci.vehicle.setVehicleClass(vehID, "passenger")
            if policy_name.startswith("StaticNumPassFL") and vehID.find("_") != -1:
                traci.vehicle.setVehicleClass(vehID, "passenger")

    if PT_LANE_SPEED_GRAPH:
        if t == 0:
            run_id = exp_name + "_" + policy_name + "_" + str(av_rate)
            wandb.init(project=exp_name, name=policy_name+"_"+str(av_rate), id=run_id)
        # calc all vehicles speed in the road
        vehIDs = traci.vehicle.getIDList()
        mean_speed = np.mean([traci.vehicle.getSpeed(vehID) for vehID in vehIDs])
        mean_speed_in_end_PTL = traci.lane.getLastStepMeanSpeed("E7_2")
        num_vehs_in_PTL = sum([traci.lane.getLastStepVehicleNumber(l) for l in traci.lane.getIDList() if len(traci.lane.getAllowed(l)) > 0])
        num_total_vehs = len(vehIDs)
        num_hdv_in_end_PTL = traci.lane.getLastStepVehicleNumber("E7_1") + traci.lane.getLastStepVehicleNumber("E7_0")
        wandb.log({"num_vehs_in_PTL": num_vehs_in_PTL, "num_total_vehs": num_total_vehs,
                   "num_hdv_in_end_PTL": num_hdv_in_end_PTL,"mean_speed": mean_speed,
                   "mean_speed_in_end_PTL": mean_speed_in_end_PTL})

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
    df["numPass"] = df["vType"].apply(lambda x: x.split("_")[1])
    df["vType"] = df["vType"].apply(lambda x: x.split("_")[0])
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


if __name__ == '__main__':
    # Example usage
    AV_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    policies = ["Nothing"]+[f"StaticNumPass_{i}" for i in range(1, 6)]
    # parse_all_output_files(AV_rates, 1, policies)
    policies.remove("Nothing")
    parse_all_pairwise(policies, AV_rates)
    # policies.remove("Nothing")
    # create_all_results_tables(AV_rates, policies)
    # STOP_FROM_RANGE = [300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
    # STOP_TO_RANGE = [0, 100, 200]
    # policies = ["DisallowBack"]
    # policy_names = [f"{policy}_{stop_from}_{stop_to}" for policy in policies for stop_from in STOP_FROM_RANGE for stop_to in STOP_TO_RANGE]
    # create_all_results_tables(AV_rates, policy_names)
