from xml.etree import ElementTree as ET

exp_name = "emergency"


def set_rou_file(flow, av_prob, emergency_prob):
    if av_prob + emergency_prob > 1:
        av_prob = 1 - emergency_prob
    # Load and parse the XML file
    tree = ET.parse(f'../{exp_name}.rou.xml')
    root = tree.getroot()

    # Find the 'MajorFlow' element
    for flow_obj in root.findall('flow'):
        if flow_obj.get('id') == 'MajorFlow':
            flow_obj.set('vehsPerHour', str(flow))
            break

    # Find the vtypes
    for vtype in root.find("vTypeDistribution").findall('vType'):
        if vtype.get('id') == 'emergency':
            vtype.set('probability', str(emergency_prob))
        elif vtype.get('id') == 'AV':
            vtype.set('probability', str(av_prob))
        elif vtype.get('id') == 'HD':
            vtype.set('probability', str(1 - av_prob - emergency_prob))

    # Save the changes back to the file
    tree.write(f'{exp_name}_flow{flow}_av{av_prob}_emer{emergency_prob}.rou.xml')

if __name__ == '__main__':
    for flow in [1000, 2000, 3000, 4000, 5000]:
        for av_prob in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            set_rou_file(flow, av_prob, emergency_prob=0.003)