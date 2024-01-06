import os
import sys

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

GUI = True

# SIM parameters
NUM_REPS = 10
SIM_DURATION = 600

# Traffic parameters
MAJOR_FLOW = 2500
AV_PROB = None # testing many AV probabilities

# Controllable parameters
DETECT_MERGING_LOC = 40
SLOW_LOC = 35
SLOW_LEN = 15
DESIRED_SLOW_SPEED = None # testing many desired slow speeds

sumoCfg = r"..\merge.sumocfg"
sumoBinary = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe" if GUI else \
    r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe"
sumoCmd = [sumoBinary, "-c", sumoCfg]