from xml.etree import ElementTree as ET

exp_name = "PublicTransportV2"


def set_rou_file(av_prob):
    # round the probabilities to 2 decimal places
    av_prob = round(av_prob, 2)
    hd_prob = round(1 - av_prob, 2)
    # Load and parse the XML file
    tree = ET.parse(f'../{exp_name}.rou.xml')
    root = tree.getroot()

    # # Find the 'MajorFlow' element
    # for flow_obj in root.findall('flow'):
    #     if flow_obj.get('id') == 'MajorFlow':
    #         flow_obj.set('vehsPerHour', str(flow))
    #         break

    # Find the vtypes
    for vtype in root.find("vTypeDistribution").findall('vType'):
        # if vtype.get('id') == 'Bus':
        #     vtype.set('probability', str(Bus_prob))
        if vtype.get('id') == 'AV':
            vtype.set('probability', str(av_prob))
        elif vtype.get('id') == 'HD':
            vtype.set('probability', str(hd_prob))



    # Save the changes back to the file
    # tree.write(f'{exp_name}_flow{flow}_av{av_prob}_Bus{Bus_prob}.rou.xml')
    tree.write(f'{exp_name}_av{av_prob}.rou.xml')

if __name__ == '__main__':
    for av_prob in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1.0]:
        set_rou_file(av_prob)