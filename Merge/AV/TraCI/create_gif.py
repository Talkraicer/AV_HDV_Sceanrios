import os
import traci
import imageio
from simulation_run import handle_step, DETECT_MERGING_LOC, SLOW_LOC, SLOW_LEN

GUI = True
sumoCfg = r"..\merge.sumocfg"
sumoBinary = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe" if GUI else \
    r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe"
sumoCmd = [sumoBinary, "-c", sumoCfg]

def record_sumo_simulation_to_gif(major_rate, desired_slow_speed=0, av_prob=0.5):
    # Start SUMO simulation
    traci.start(sumoCmd)

    frames = []

    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        handle_step(step, av_prob, DETECT_MERGING_LOC, SLOW_LOC, SLOW_LEN, desired_slow_speed)
        image_file = f"frame_{step}.png"
        # Capture the current state of the simulation as an image
        traci.gui.screenshot("View #0", image_file)
        traci.simulationStep()
        step += 1
    traci.close()
    for s in range(step):
        frame_name = f"frame_{s}.png"
        frames.append(imageio.imread(frame_name))
        os.remove(frame_name)
    # Create a GIF from the captured frames
    imageio.mimsave(f"{major_rate}Major_{desired_slow_speed}DSS_{av_prob}AVprob.gif", frames, fps=10)


# Example usage
record_sumo_simulation_to_gif(2500)
