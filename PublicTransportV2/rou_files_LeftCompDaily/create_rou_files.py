from xml.etree import ElementTree as ET
import numpy as np
import scipy.stats as stats

exp_name = "LeftCompDaily"
PROB_PASS_AV = {0: 0.3, 1: 0.3, 2: 0.2, 3: 0.1, 4: 0.07, 5: 0.03} # Expectation = 1.1
PROB_PASS_HD = {1: 0.8, 2: 0.1, 3: 0.05, 4: 0.03, 5: 0.02} # Expectation = 1.37

VEH_AMOUNT = {6:6163, 7:6450,8:7053,9:6443,10:6287,11:5800,12:6266,13:5428,
              14:5661,15:4644,16:4937,17:5668,18:5184,19:5126}
EXIT_PROP = 0.15
BUS_AMOUNT = {6:62, 7:37,8:19,9:31,10:26,11:25,12:17,13:31,
              14:44,15:30,16:24,17:28,18:25,19:16}

BUS_PASS_RANGE = range(25, 45)
PROB_PASS_BUS = {i: 1 / len(BUS_PASS_RANGE) for i in BUS_PASS_RANGE} # Expectation = 25


def set_rou_file(av_prob):
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
        flow = ET.Element('flow', id=f'MajorFlow{hour}', type="vehicleDist", begin=str((hour-6) * 3600), departLane="random",
                          fromJunction="J0", toJunction="J9",end=str((hour -5) * 3600), vehsPerHour=str(VEH_AMOUNT[hour]*(1-3*EXIT_PROP)), departSpeed="max")
        flow_J3 = ET.Element('flow', id=f'MajorFlow{hour}_J3', type="vehicleDist", begin=str((hour-6) * 3600), departLane="random",
                          fromJunction="J0", toJunction="J3",end=str((hour -5) * 3600), vehsPerHour=str(int(VEH_AMOUNT[hour]* EXIT_PROP)), departSpeed="max", arrivalLane="0")
        flow_J5 = ET.Element('flow', id=f'MajorFlow{hour}_J5', type="vehicleDist", begin=str((hour-6) * 3600), departLane="random",
                          fromJunction="J0", toJunction="J5",end=str((hour -5) * 3600), vehsPerHour=str(int(VEH_AMOUNT[hour]* EXIT_PROP)), departSpeed="max", arrivalLane="0")
        flow_J7 = ET.Element('flow', id=f'MajorFlow{hour}_J7', type="vehicleDist", begin=str((hour-6) * 3600), departLane="random",
                          fromJunction="J0", toJunction="J7",end=str((hour -5) * 3600), vehsPerHour=str(int(VEH_AMOUNT[hour]* EXIT_PROP)), departSpeed="max", arrivalLane="0")

        flow_J4 = ET.Element('flow', id=f'MajorFlow{hour}_J4', type="vehicleDist", begin=str((hour-6) * 3600), departLane="0",
                          fromJunction="J4", toJunction="J9",end=str((hour -5) * 3600), vehsPerHour=str(int(VEH_AMOUNT[hour]*EXIT_PROP)), departSpeed="max")
        flow_J6 = ET.Element('flow', id=f'MajorFlow{hour}_J6', type="vehicleDist", begin=str((hour-6) * 3600), departLane="0",
                            fromJunction="J6", toJunction="J9",end=str((hour -5) * 3600), vehsPerHour=str(int(VEH_AMOUNT[hour]*EXIT_PROP)), departSpeed="max")

        flow_Bus = ET.Element('flow', id=f'busFlow{hour}', type="busDist", begin=str((hour-6) * 3600), departLane="random",
                          fromJunction="J0", toJunction="J9",end=str((hour -5) * 3600), vehsPerHour=str(BUS_AMOUNT[hour]), departSpeed="max")

        for elem in [flow, flow_J3, flow_J5, flow_J7, flow_J4, flow_J6, flow_Bus]:
            elem.tail = '\n\t'
            root.append(elem)

    # Save the changes back to the file
    # tree.write(f'{exp_name}_flow{flow}_av{av_prob}_Bus{Bus_prob}.rou.xml')
    tree.write(f'{exp_name}_av{av_prob}.rou.xml')


if __name__ == '__main__':
    for av_prob in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        set_rou_file(av_prob)