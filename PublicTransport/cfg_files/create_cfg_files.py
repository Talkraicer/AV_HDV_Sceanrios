from xml.etree import ElementTree as ET
import numpy as np

exp_name = "PublicTransport"

def set_cfg_file(flow, av_prob, Bus_prob, seed, DL=False):
    if av_prob + Bus_prob > 1:
        av_prob = 1 - Bus_prob
    # Load and parse the XML file
    tree = ET.parse(f'../{exp_name}.sumocfg')
    root = tree.getroot()

    # Find the 'MajorFlow' element
    route_file = root.find("input").find('route-files')
    route_file.set('value', f'../rou_files/{exp_name}_flow{flow}_av{av_prob}_Bus{Bus_prob}.rou.xml')

    seed_element = root.find("random_number").find('seed')
    seed_element.set('value', str(seed))

    # Save the changes back to the file
    if DL:
        tree.write(f"../cfg_files_DL/{exp_name}_flow{flow}_av{av_prob}_Bus{Bus_prob}.sumocfg")
    else:
        tree.write(f'{exp_name}_flow{flow}_av{av_prob}_Bus{Bus_prob}.sumocfg')

if __name__ == '__main__':
    for flow in [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]:
        seed = np.random.randint(0, 10000)
        for av_prob in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            set_cfg_file(flow, av_prob, Bus_prob=0.1, seed=seed)
            set_cfg_file(flow, av_prob, Bus_prob=0.1, seed=seed, DL=True)