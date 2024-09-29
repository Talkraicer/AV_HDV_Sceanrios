import os

file_types = [ ".net.xml", ".rou.xml", ".sumocfg"]


def duplicate_experiment(input_exp_name, input_exp_short_name, output_exp_name):
    for file_type in file_types:
        input_file = f'{input_exp_name}{file_type}'
        output_file = f'{output_exp_name}{file_type}'
        with open(input_file, 'r') as f:
            if file_type == ".sumocfg":
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if "route-files" in line:
                        lines[i] = f'    <route-files value="{output_exp_name}.rou.xml"/>\n'
                    elif "additional-files" in line:
                        lines[i] = f'    <additional-files value="{output_exp_name}.add.xml"/>\n'
                    elif "net-file" in line:
                        lines[i] = f'    <net-file value="{output_exp_name}.net.xml"/>\n'
                with open(output_file, 'w') as f2:
                    f2.writelines(lines)
            else:
                with open(output_file, 'w') as f2:
                    f2.write(f.read())
    os.mkdir(f"cfg_files_{output_exp_name}")
    with open(f"cfg_files_{input_exp_short_name}/create_cfg_files.py", 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if "exp_name =" in line:
                lines[i] = f'exp_name = "{output_exp_name}"\n'
            if "route_file.set" in line:
                lines[i] = f'    route_file.set(\'value\', f\'../rou_files_{output_exp_name}/{output_exp_name}_av{{av_prob}}.rou.xml\')\n'
        with open(f"cfg_files_{output_exp_name}/create_cfg_files.py", 'w') as f2:
            f2.writelines(lines)
    os.mkdir(f"rou_files_{output_exp_name}")
    with open(f"rou_files_{input_exp_short_name}/create_rou_files.py", 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if "exp_name =" in line:
                lines[i] = f'exp_name = "{output_exp_name}"\n'
        with open(f"rou_files_{output_exp_name}/create_rou_files.py", 'w') as f2:
            f2.writelines(lines)
    # os.system(f"cd cfg_files_{output_exp_name}")
    # os.system(f"python cfg_files_{output_exp_name}/create_cfg_files.py")
    # os.system(f"cd ../rou_files_{output_exp_name}")
    # os.system(f"python rou_files_{output_exp_name}/create_rou_files.py")

if __name__ == '__main__':
    duplicate_experiment("LeftCompScenarios", "LeftCompScenarios", "RandomLeftCompScenarios")
