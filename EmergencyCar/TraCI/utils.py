import os
import traci
import imageio
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

exp_name = "emergency"
NUM_REPS = 1
GUI = True
sumoCfg = fr"..\{exp_name}.sumocfg"
sumoBinary = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe" if GUI else \
    r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe"
sumoCmd = [sumoBinary, "-c", sumoCfg]
metrics = ["duration", "departDelay", "speed", "timeLoss", "totalDelay"]

def add_emergency_car(id, depart_time):
    traci.vehicle.add(f"emergencyCar{id}", routeID="r_0", typeID="emergency", depart=str(depart_time),
                      departLane="random")


def handle_step(t, av_prob, emergency_prob, policy_name, randomizer):
    vehIDs = traci.vehicle.getIDList()
    has_emergency = False
    for vehID in vehIDs:
        if traci.vehicle.getTypeID(vehID) == "DEFAULT_VEHTYPE":
            rand = randomizer.uniform(0, 1)
            if rand < av_prob:
                traci.vehicle.setType(vehID, "AV")
            elif rand < emergency_prob + av_prob:
                traci.vehicle.setType(vehID, "emergency")
                traci.vehicle.setSpeedFactor(vehID, 3)
                traci
            else:
                traci.vehicle.setType(vehID, "HD")
        speed, lane, lanePos = traci.vehicle.getSpeed(vehID), traci.vehicle.getLaneID(vehID), \
            traci.vehicle.getLanePosition(vehID)
        if traci.vehicle.getTypeID(vehID) == "emergency":
            has_emergency = True
            if policy_name == "ClearFront":
                # clear all vehicles in front of the emergency vehicle
                leader = traci.vehicle.getLeader(vehID, 0)
                while leader:
                    frontVehID, dist = leader
                    if traci.vehicle.getTypeID(frontVehID) == "AV" and traci.vehicle.getLaneID(frontVehID) == lane:
                        to_lane = 1 if lane.endswith("2") else 0
                        traci.vehicle.changeLane(frontVehID, to_lane, 1)
                    leader = traci.vehicle.getLeader(frontVehID, 0)
    if policy_name == "ClearLeft" or policy_name == "ClearLeftWhenEmergency":
        for vehID in vehIDs:
            if traci.vehicle.getTypeID(vehID) == "AV":
                if policy_name == "ClearLeft":
                    if lane.endswith("2"):
                        traci.vehicle.changeLane(vehID, 1, 1)
                elif policy_name == "ClearLeftWhenEmergency":
                    if lane.endswith("2") and has_emergency:
                        traci.vehicle.changeLane(vehID, 1, 1)
    return has_emergency


def set_MajorFlow_vehsPerHour(vehsPerHour):
    # Load and parse the XML file
    tree = ET.parse(f'../{exp_name}.rou.xml')
    root = tree.getroot()

    # Find the 'MajorFlow' element
    for flow in root.findall('flow'):
        if flow.get('id') == 'MajorFlow':
            flow.set('vehsPerHour', str(vehsPerHour))
            break

    # Save the changes back to the file
    tree.write(f'../{exp_name}.rou.xml')


def set_sim_duration(sim_duration):
    # Load and parse the XML file
    tree = ET.parse(f'../{exp_name}.rou.xml')
    root = tree.getroot()

    # Find the 'MajorFlow' element
    for flow in root.findall('flow'):
        flow.set('end', str(sim_duration))

    # Save the changes back to the file
    tree.write(f'../{exp_name}.rou.xml')


def set_sumo_simulation(major_rate, sim_duration):
    set_MajorFlow_vehsPerHour(major_rate)
    set_sim_duration(sim_duration)


def record_sumo_simulation_to_gif(major_rate, policy_name, record_duration=180, desired_slow_speed=0, av_prob=0.5):
    # Start SUMO simulation
    traci.start(sumoCmd)

    frames = []

    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        handle_step(step, av_prob, policy_name)
        if "frames" not in os.listdir():
            os.mkdir("frames")
        image_file = f"frames/frame_{step}.png"
        # Capture the current state of the simulation as an image
        traci.gui.screenshot("View #0", image_file)
        traci.simulationStep()
        step += 1
        if step > record_duration:
            break
    traci.close(False)
    for s in range(step):
        frame_name = f"frames/frame_{s}.png"
        frames.append(imageio.imread(frame_name))
        os.remove(frame_name)
    # Create a GIF from the captured frames
    imageio.mimsave(
        f"results_gifs/{exp_name}_{major_rate}Major_{desired_slow_speed}DSS_{av_prob}AVprob_{policy_name}.gif", frames,
        fps=10)


def output_file_to_df(output_file, num_reps):
    # Parse the XML file into pd dataframe
    tree = ET.parse(output_file)
    root = tree.getroot()

    dict = {"duration": [], "departDelay": [], "routeLength": [], "vType": [], "timeLoss": []}
    for tripinfo in root.findall('tripinfo'):
        for key in dict.keys():
            dict[key].append(tripinfo.get(key))
    df = pd.DataFrame(dict)
    df["speed"] = df.routeLength.astype(float) / df.duration.astype(float)
    df["totalDelay"] = df.departDelay.astype(float) + df.timeLoss.astype(float)
    df.drop(columns=["routeLength"], inplace=True)
    # convert to float except vType
    for col in df.columns:
        if col != "vType":
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


def calc_stats(df):
    # Calculate statistics per vType
    stats = {}
    for vType in df.vType.unique():
        df_vType = df[df.vType == vType]
        stats[vType] = {}
        for metric in metrics:
            stats[vType][f"avg_{metric}"] = df_vType[metric].mean()
            stats[vType][f"std_{metric}"] = df_vType[metric].std(ddof=1)
    if "all" not in stats.keys():
        stats["all"] = {}
        for metric in metrics:
            stats["all"][f"avg_{metric}"] = df[metric].mean()
            stats["all"][f"std_{metric}"] = df[metric].std(ddof=1)
    return pd.DataFrame(stats)


def parse_output_files(av_rates, num_reps, policy_name, flow):
    # Aggregate all output files into one dataframe, divided by vType

    # set MultiIndex for df - each vType will be a column in df with all the stats
    stats_names = [f"avg_{metric}" for metric in metrics] + [f"std_{metric}" for metric in metrics]
    vType_names = ["AV", "HD", "emergency", "all"]
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


if __name__ == '__main__':
    # Example usage
    for major_flow in [1000, 2000, 3000, 4000, 5000]:
        av_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for policy_name in ["ClearFront", "Nothing"]:
            parse_output_files(av_rates, policy_name=policy_name, num_reps=NUM_REPS, flow=major_flow)
