import os
import traci
import imageio
from random import random
import xml.etree.ElementTree as ET
from simulation_run import DETECT_MERGING_LOC, SLOW_LOC, SLOW_LEN

exp_name = "merge"
GUI = True
sumoCfg = fr"..\{exp_name}.sumocfg"
sumoBinary = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe" if GUI else \
    r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe"
sumoCmd = [sumoBinary, "-c", sumoCfg]

MajorFlow_vehsPerHour = 2500


def handle_step(t, av_prob, detect_merging_loc, slow_loc, slow_len, desired_slow_speed):
    vehIDs = traci.vehicle.getIDList()
    merging = False
    for vehID in vehIDs:
        if traci.vehicle.getTypeID(vehID) == "DEFAULT_VEHTYPE":
            traci.vehicle.setType(vehID, "AV" if random() < av_prob else "HD")
        speed, lane, lanePos = traci.vehicle.getSpeed(vehID), traci.vehicle.getLaneID(vehID), \
            traci.vehicle.getLanePosition(vehID)
        if lane == "E2_0" and lanePos > detect_merging_loc:
            merging = True
        # print(vehID, ": ", lane, " SPEED:", speed, " LANEPOS:", lanePos, " TYPE:", traci.vehicle.getTypeID(vehID))
    if merging:
        for vehID in vehIDs:
            if traci.vehicle.getLaneID(vehID) == "E0_0" and traci.vehicle.getTypeID(vehID) == "AV" and \
                    slow_loc < traci.vehicle.getLanePosition(vehID) < slow_loc + slow_len:
                # Slow down the vehicle to specific period of time
                traci.vehicle.slowDown(vehID, desired_slow_speed, 1)

    return True


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


def record_sumo_simulation_to_gif(major_rate, record_duration=180, desired_slow_speed=0, av_prob=0.5):
    # Start SUMO simulation
    traci.start(sumoCmd)

    frames = []

    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        handle_step(step, av_prob, DETECT_MERGING_LOC, SLOW_LOC, SLOW_LEN, desired_slow_speed)
        if "frames" not in os.listdir():
            os.mkdir("frames")
        image_file = f"frames/frame_{step}.png"
        # Capture the current state of the simulation as an image
        traci.gui.screenshot("View #0", image_file)
        traci.simulationStep()
        step += 1
        if step > record_duration:
            break
    traci.close()
    for s in range(step):
        frame_name = f"frames/frame_{s}.png"
        frames.append(imageio.imread(frame_name))
        os.remove(frame_name)
    # Create a GIF from the captured frames
    imageio.mimsave(f"{major_rate}Major_{desired_slow_speed}DSS_{av_prob}AVprob.gif", frames, fps=10)


if __name__ == '__main__':
    # Example usage
    record_sumo_simulation_to_gif(MajorFlow_vehsPerHour)
