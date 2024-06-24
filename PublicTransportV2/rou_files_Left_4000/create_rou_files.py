from xml.etree import ElementTree as ET
import numpy as np
import scipy.stats as stats

exp_name = "Left_4000"
PROB_PASS_AV = {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.15, 4: 0.15, 5: 0.1} # Expectation = 1.75
PROB_PASS_HD = {1: 0.7, 2: 0.2, 3: 0.05, 4: 0.03, 5: 0.02} # Expectation = 1.47
BUS_PASS_RANGE = range(15, 35)
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

    # Save the changes back to the file
    # tree.write(f'{exp_name}_flow{flow}_av{av_prob}_Bus{Bus_prob}.rou.xml')
    tree.write(f'{exp_name}_av{av_prob}.rou.xml')


if __name__ == '__main__':
    for av_prob in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        set_rou_file(av_prob)
