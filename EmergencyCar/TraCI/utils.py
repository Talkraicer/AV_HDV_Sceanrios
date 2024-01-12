import os
import traci
import imageio
from random import random
import xml.etree.ElementTree as ET

exp_name = "emergency"

GUI = True
sumoCfg = fr"..\{exp_name}.sumocfg"
sumoBinary = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe" if GUI else \
    r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe"
sumoCmd = [sumoBinary, "-c", sumoCfg]

# Parameters that should be given from simulation_run.py
MajorFlow_vehsPerHour = 5000
sim_duration = 300

def handle_step(t, av_prob, policy_name = "Nothing"):
    vehIDs = traci.vehicle.getIDList()
    has_emergency = False
    for vehID in vehIDs:
        if traci.vehicle.getTypeID(vehID) == "DEFAULT_VEHTYPE":
            traci.vehicle.setType(vehID, "AV" if random() < av_prob else "HD")
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
    imageio.mimsave(f"results_gifs/{exp_name}_{major_rate}Major_{desired_slow_speed}DSS_{av_prob}AVprob_{policy_name}.gif", frames, fps=10)


if __name__ == '__main__':
    # Example usage
    set_sumo_simulation(MajorFlow_vehsPerHour, sim_duration)
    record_sumo_simulation_to_gif(MajorFlow_vehsPerHour,"Nothing")
