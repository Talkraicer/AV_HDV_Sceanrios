exp_name = "LeftCompScenarios"

import os
import traci
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import wandb
import time
from results_utils import output_file_to_df, calc_stats_metric
from log_utils import log_features, init_wandb_logger

NUM_PROCESSES = 70
GUI = False
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
LOG_RATE = 100  # Switch to zero for no logging
DELETE_OLDER = True

# Control Var Min Start
CONTROL_MIN_START = 1

# Clipping parameters:
NUM_VEHS_PTL_MIN = 10
NUM_VEHS_PTL_MAX = 60



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


def allow_min_pass(policy_name, control_min_start):
    vehIDs = traci.vehicle.getIDList()
    for vehID in vehIDs:
        typeID = traci.vehicle.getTypeID(vehID)
        if typeID.startswith("AV") and int(typeID.split("_")[1][0]) >= control_min_start:
            if policy_name.startswith("Control") or policy_name.startswith("StaticNumPassFL"):
                loc = traci.vehicle.getPosition(vehID)[0]
                if loc < 300:
                    traci.vehicle.setVehicleClass(vehID, "private")
            else:
                traci.vehicle.setVehicleClass(vehID, "private")


def handle_step(t, policy_name, av_rate, log_rate=LOG_RATE):
    global BUSES_VOLUNTEERS
    if policy_name == "Nothing" and exp_name.startswith("Left") and t < 1:
        for lane in traci.lane.getIDList():
            if "bus" in traci.lane.getAllowed(lane):
                traci.lane.setAllowed(lane, "bus")

    if policy_name.startswith("Plus"):
        control_min_start = int(policy_name.split("_")[1])
        vehIDs = traci.vehicle.getIDList()
        for vehID in vehIDs:
            typeID = traci.vehicle.getTypeID(vehID)
            if int(typeID.split("_")[1][0]) >= control_min_start:
                traci.vehicle.setVehicleClass(vehID, "private")
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

    global CONTROL_MIN_START
    if policy_name.startswith("StaticNumPass"):
        min_num_pass = int(policy_name.split("_")[1][0])
        CONTROL_MIN_START = min_num_pass
        allow_min_pass(policy_name, CONTROL_MIN_START)

    if log_rate and t % log_rate == 0:
        if t == 0:
            init_wandb_logger(policy_name, av_rate, delete_older=DELETE_OLDER)

        log_msg = log_features(policy_name + exp_name + "_" + str(av_rate) + ".xml", t, LOG_RATE)

        if policy_name.startswith("Control") and log_msg:
            control_var = policy_name.split()[1]
            control_var_min = int(policy_name.split()[2])
            control_var_max = int(policy_name.split()[3])
            changed = False
            if "Clipped" in policy_name:
                num_vehs_in_PTL = log_msg["num_vehs_in_PTL"]
                if num_vehs_in_PTL < NUM_VEHS_PTL_MIN:
                    CONTROL_MIN_START -= 1
                    changed = True
                elif num_vehs_in_PTL > NUM_VEHS_PTL_MAX:
                    CONTROL_MIN_START += 1
                    changed = True
            if not changed:
                if log_msg[control_var] < control_var_min:
                    CONTROL_MIN_START += 1
                elif log_msg[control_var] > control_var_max:
                    CONTROL_MIN_START -= 1
            CONTROL_MIN_START = max(1, CONTROL_MIN_START)
            CONTROL_MIN_START = min(6, CONTROL_MIN_START)
        if log_msg:
            log_msg["MinPassNum"] = CONTROL_MIN_START
            wandb.log(log_msg)
    if policy_name.startswith("Control"):
        allow_min_pass(policy_name, CONTROL_MIN_START)
