from xml.etree import ElementTree as ET
import numpy as np
import scipy.stats as stats
import copy
import pandas as pd
import matplotlib.pyplot as plt
def normalize_dict(d):
    total = sum(d.values())
    return {k: v / total for k, v in d.items()}

exp_name = "LeftCompScenarios"
PROB_PASS_HD = {1: 0.63, 2: 0.28, 3: 0.06, 4: 0.02, 5: 0.01}
FACTOR_AV = 1
PROB_PASS_AV = copy.deepcopy(PROB_PASS_HD)
PROB_PASS_AV[1] *= FACTOR_AV
PROB_PASS_AV = normalize_dict(PROB_PASS_AV)

print("Expected number of passengers in AVs: ", sum([k*v for k,v in PROB_PASS_AV.items()]))
print("Expected number of passengers in HDs: ", sum([k*v for k,v in PROB_PASS_HD.items()]))


VEH_AMOUNT = {
              6: 4000,7:7000,8:7000,9:4000, # RUSH HOURS
              10:0,11:0, # BREAK
              12:9000, # PEAK
              13:0,14:0, # BREAK
              15:5800,16:5800, #MID DAY
              17:0,18:0, # BREAK
              19:4000,20:4000, # WEEKEND
              21:0,22:0,# BREAK
              23:4000, 24: 5000, 25:6000, 26:7000, 27:6000, 28: 5000, 29:4000 # Moderate Peak
              }
df = pd.DataFrame(VEH_AMOUNT.items(), columns = ['Hour', 'Vehicles'])
df.plot(x = 'Hour', y = 'Vehicles', kind = 'bar')
plt.suptitle('Vehicles demand per hour')
plt.grid()
plt.show()
EXIT_PROP = 0.1
BUS_AMOUNT = {
              6: 30,7:60,8:60,9:30, # RUSH HOURS
              10:0,11:0, # BREAK
              12:40, # PEAK
              13:0,14:0, # BREAK
              15:40,16:40, #MID DAY
              17:0,18:0, # BREAK
              19:0,20:0, # WEEKEND
              21:0,22:0,# BREAK
              23:30,24: 40, 25:50, 26:60, 27:50, 28: 40, 29:30 # Moderate Peak
              }


# TODO: Find bus occupancy distribution
BUS_PASS_RANGE = range(25, 45)
PROB_PASS_BUS = {i: 1 / len(BUS_PASS_RANGE) for i in BUS_PASS_RANGE} # Expectation = 25


def set_rou_file(av_prob, HOUR_LEN = 3600):
    # round the probabilities to 2 decimal places
    av_prob = round(av_prob, 2)
    hd_prob = round(1 - av_prob, 2)
    # Load and parse the XML file
    tree = ET.parse(f'../{exp_name}.rou.xml')
    root = tree.getroot()

    # Set vTypeDistribution to contain the probabilities of each vehicle type and the number of passengers
    for vTypeDist in root.findall('vTypeDistribution'):
        vTypeDist.text += '\t'
        if vTypeDist.attrib['id'] == 'vehicleDist':
            for k,v in PROB_PASS_AV.items():
                prob = round(av_prob * v, 5)
                elem = ET.Element('vType', id=f'AV_{k}', color='blue', probability=str(prob), vClass='evehicle')
                elem.tail = '\n\t\t'
                vTypeDist.append(elem)
            for k,v in PROB_PASS_HD.items():
                prob = round(hd_prob * v,5)
                elem = ET.Element('vType', id=f'HD_{k}', color='red', probability=str(prob), vClass='passenger')
                elem.tail = '\n\t\t'
                vTypeDist.append(elem)
        elif vTypeDist.attrib['id'] == 'busDist':
            for k,v in PROB_PASS_BUS.items():
                elem = ET.Element('vType', id=f'Bus_{k}', probability=str(v), vClass='bus')
                elem.tail = '\n\t\t'
                vTypeDist.append(elem)

    # Create a flow for each hour of the day
    for hour in VEH_AMOUNT.keys():
        if VEH_AMOUNT[hour] == 0:
            continue
        flow = ET.Element('flow', id=f'MajorFlow{hour}', type="vehicleDist", begin=str((hour-6) * HOUR_LEN), departLane="random",
                          fromJunction="J0", toJunction="J9",end=str((hour -5) * HOUR_LEN), vehsPerHour=str(int(VEH_AMOUNT[hour]*(1-2*EXIT_PROP))), departSpeed="max")
        flow_J3 = ET.Element('flow', id=f'MajorFlow{hour}_J3', type="vehicleDist", begin=str((hour-6) * HOUR_LEN), departLane="random",
                          fromJunction="J0", toJunction="J3",end=str((hour -5) * HOUR_LEN), vehsPerHour=str(int(VEH_AMOUNT[hour]* EXIT_PROP)), departSpeed="max", arrivalLane="0")
        flow_J5 = ET.Element('flow', id=f'MajorFlow{hour}_J5', type="vehicleDist", begin=str((hour-6) * HOUR_LEN), departLane="random",
                          fromJunction="J0", toJunction="J5",end=str((hour -5) * HOUR_LEN), vehsPerHour=str(int(VEH_AMOUNT[hour]* EXIT_PROP)), departSpeed="max", arrivalLane="0")
        flow_J7 = ET.Element('flow', id=f'MajorFlow{hour}_J7', type="vehicleDist", begin=str((hour-6) * HOUR_LEN), departLane="random",
                          fromJunction="J0", toJunction="J7",end=str((hour -5) * HOUR_LEN), vehsPerHour=str(int(VEH_AMOUNT[hour]* EXIT_PROP)), departSpeed="max", arrivalLane="0")

        flow_J4 = ET.Element('flow', id=f'MajorFlow{hour}_J4', type="vehicleDist", begin=str((hour-6) * HOUR_LEN), departLane="0",
                          fromJunction="J4", toJunction="J9",end=str((hour -5) * HOUR_LEN), vehsPerHour=str(int(VEH_AMOUNT[hour]*EXIT_PROP)), departSpeed="max")
        flow_J2 = ET.Element('flow', id=f'MajorFlow{hour}_J2', type="vehicleDist", begin=str((hour-6) * HOUR_LEN), departLane="0",
                             fromJunction="J2", toJunction="J9",end=str((hour -5) * HOUR_LEN), vehsPerHour=str(int(VEH_AMOUNT[hour]*EXIT_PROP)), departSpeed="max")
        flow_J6 = ET.Element('flow', id=f'MajorFlow{hour}_J6', type="vehicleDist", begin=str((hour-6) * HOUR_LEN), departLane="0",
                            fromJunction="J6", toJunction="J9",end=str((hour -5) * HOUR_LEN), vehsPerHour=str(int(VEH_AMOUNT[hour]*EXIT_PROP)), departSpeed="max")
        if BUS_AMOUNT[hour] != 0:
            flow_Bus = ET.Element('flow', id=f'busFlow{hour}', type="busDist", begin=str((hour-6) * HOUR_LEN), departLane="random",
                              fromJunction="J0", toJunction="J9",end=str((hour -5) * HOUR_LEN), vehsPerHour=str(BUS_AMOUNT[hour]), departSpeed="max")

        for elem in [flow, flow_J3, flow_J5, flow_J7, flow_J2, flow_J4, flow_J6]:
            elem.tail = '\n\t'
            root.append(elem)
        if BUS_AMOUNT[hour] != 0:
            flow_Bus.tail = '\n\t'
            root.append(flow_Bus)

    # Save the changes back to the file
    # tree.write(f'{exp_name}_flow{flow}_av{av_prob}_Bus{Bus_prob}.rou.xml')
    tree.write(f'{exp_name}_av{av_prob}.rou.xml')


if __name__ == '__main__':
    for av_prob in [0.1,0.2,0.3,0.4,0.6,0.8]:
        set_rou_file(av_prob, HOUR_LEN = 1800)

    set_rou_file(0.5, HOUR_LEN = 15)
