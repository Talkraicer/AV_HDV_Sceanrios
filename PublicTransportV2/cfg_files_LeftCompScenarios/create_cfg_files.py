from xml.etree import ElementTree as ET
import numpy as np

np.random.seed(42)

exp_name = "LeftCompScenarios"

def set_cfg_file(av_prob, seed):
    # Load and parse the XML file
    tree = ET.parse(f'../{exp_name}.sumocfg')
    root = tree.getroot()

    # set route file
    route_file = root.find("input").find('route-files')
    route_file.set('value', f'../rou_files_LeftCompScenarios/LeftCompScenarios_av{av_prob}.rou.xml')



    # set net file
    net_file = root.find("input").find('net-file')
    net_file.set('value', f'../{exp_name}.net.xml')

    # set seed
    seed_element = root.find("random_number").find('seed')
    seed_element.set('value', str(seed))

    # Save the changes back to the file
    tree.write(f'{exp_name}_av{av_prob}.sumocfg')

if __name__ == '__main__':
    seed = np.random.randint(0, 10000)
    for av_prob in [0.1,0.2,0.3,0.4,0.6,0.8]:
        set_cfg_file(av_prob, seed)
