from xml.etree import ElementTree as ET
import numpy as np
import scipy.stats as stats
import copy
import pandas as pd
import matplotlib.pyplot as plt
def normalize_dict(d):
    total = sum(d.values())
    return {k: v / total for k, v in d.items()}

exp_name = "RandomLeftCompDaily"
PROB_PASS_HD = {1: 0.63, 2: 0.28, 3: 0.06, 4: 0.02, 5: 0.01}
FACTOR_AV = 1
PROB_PASS_AV = copy.deepcopy(PROB_PASS_HD)
PROB_PASS_AV[1] *= FACTOR_AV
PROB_PASS_AV = normalize_dict(PROB_PASS_AV)

print("Expected number of passengers in AVs: ", sum([k*v for k,v in PROB_PASS_AV.items()]))
print("Expected number of passengers in HDs: ", sum([k*v for k,v in PROB_PASS_HD.items()]))



VEH_AMOUNT = {6:6163, 7:6450,8:7053,9:6443,10:6287,11:5800,12:6266,13:5428,
              14:5661,15:4644,16:4937,17:5668,18:5184,19:5126}
EXIT_PROP = 0.1
BUS_AMOUNT = {6:62, 7:37,8:19,9:31,10:26,11:25,12:17,13:31,
              14:44,15:30,16:24,17:28,18:25,19:16}

df = pd.DataFrame(VEH_AMOUNT.items(), columns = ['Hour', 'Vehicles'])
df.plot(x = 'Hour', y = 'Vehicles', kind = 'bar')
plt.suptitle('Vehicles demand per hour')
plt.grid()
plt.show()


# TODO: Find bus occupancy distribution
BUS_PASS_RANGE = range(25, 45)
PROB_PASS_BUS = {i: 1 / len(BUS_PASS_RANGE) for i in BUS_PASS_RANGE} # Expectation = 25

NUM_LANES = 4 # Number of lanes in the beginning of the road

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

        elements = []
        # insertion - different flow for each lane and hour
        for lane in range(NUM_LANES):
            main_v_prob = VEH_AMOUNT[hour] * (1 - 3 * EXIT_PROP) / float(NUM_LANES * 3600)
            exit_v_prob = VEH_AMOUNT[hour]* EXIT_PROP / float(NUM_LANES * 3600)
            lane_name = str(lane)
            flow = ET.Element('flow', id=f'MajorFlow{hour}_{lane_name}', type="vehicleDist", begin=str((hour-6) * HOUR_LEN), departLane=lane_name,
                              fromJunction="J0", toJunction="J9",end=str((hour -5) * HOUR_LEN), period=f"exp({main_v_prob})", departSpeed="max")
            flow_J3 = ET.Element('flow', id=f'MajorFlow{hour}_J3_{lane_name}', type="vehicleDist", begin=str((hour-6) * HOUR_LEN), departLane=lane_name,
                              fromJunction="J0", toJunction="J3",end=str((hour -5) * HOUR_LEN), period=f"exp({exit_v_prob})", departSpeed="max", arrivalLane="0")
            flow_J5 = ET.Element('flow', id=f'MajorFlow{hour}_J5_{lane_name}', type="vehicleDist", begin=str((hour-6) * HOUR_LEN), departLane=lane_name,
                              fromJunction="J0", toJunction="J5",end=str((hour -5) * HOUR_LEN), period=f"exp({exit_v_prob})", departSpeed="max", arrivalLane="0")
            flow_J7 = ET.Element('flow', id=f'MajorFlow{hour}_J7_{lane_name}', type="vehicleDist", begin=str((hour-6) * HOUR_LEN), departLane=lane_name,
                              fromJunction="J0", toJunction="J7",end=str((hour -5) * HOUR_LEN), period=f"exp({exit_v_prob})", departSpeed="max", arrivalLane="0")
            elements.append(flow)
            elements.append(flow_J3)
            elements.append(flow_J5)
            elements.append(flow_J7)

        v_prob = VEH_AMOUNT[hour]*EXIT_PROP / float(HOUR_LEN)
        flow_J4 = ET.Element('flow', id=f'MajorFlow{hour}_J4', type="vehicleDist", begin=str((hour-6) * HOUR_LEN), departLane="0",
                          fromJunction="J4", toJunction="J9",end=str((hour -5) * HOUR_LEN), period=f"exp({v_prob})", departSpeed="max")
        flow_J2 = ET.Element('flow', id=f'MajorFlow{hour}_J2', type="vehicleDist", begin=str((hour-6) * HOUR_LEN), departLane="0",
                             fromJunction="J2", toJunction="J9",end=str((hour -5) * HOUR_LEN), period=f"exp({v_prob})", departSpeed="max")
        flow_J6 = ET.Element('flow', id=f'MajorFlow{hour}_J6', type="vehicleDist", begin=str((hour-6) * HOUR_LEN), departLane="0",
                            fromJunction="J6", toJunction="J9",end=str((hour -5) * HOUR_LEN), period=f"exp({v_prob})", departSpeed="max")
        elements.append(flow_J2)
        elements.append(flow_J4)
        elements.append(flow_J6)

        if BUS_AMOUNT[hour] != 0:
            flow_Bus = ET.Element('flow', id=f'busFlow{hour}', type="busDist", begin=str((hour-6) * HOUR_LEN), departLane="random",
                              fromJunction="J0", toJunction="J9",end=str((hour -5) * HOUR_LEN), vehsPerHour=str(BUS_AMOUNT[hour]), departSpeed="max")

        for elem in elements:
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
        set_rou_file(av_prob, HOUR_LEN = 3600)