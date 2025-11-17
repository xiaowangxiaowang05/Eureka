import yaml

# # Load the YAML file
# task = 'Cartpole'
# suffix = 'GPT'

def create_task(root_dir, task, env_name, suffix):
    # Create task YAML file 
    input_file = f"{root_dir}/cfg/task/{task}.yaml"
    output_file = f"{root_dir}/cfg/task/{task}{suffix}.yaml"
    with open(input_file, 'r') as yamlfile:
        data = yaml.safe_load(yamlfile)

    # Modify the "name" field
    data['name'] = f'{task}{suffix}'
    data['env']['env_name'] = f'{env_name}{suffix}'
    
    # Write the new YAML file
    with open(output_file, 'w') as new_yamlfile:
        yaml.safe_dump(data, new_yamlfile)

    # Create training YAML file
    input_file = f"{root_dir}/cfg/train/{task}PPO.yaml"
    output_file = f"{root_dir}/cfg/train/{task}{suffix}PPO.yaml"

    with open(input_file, 'r') as yamlfile:
        data = yaml.safe_load(yamlfile)

    # Modify the "name" field
    data['params']['config']['name'] = data['params']['config']['name'].replace(task, f'{task}{suffix}')

    # Add mini_epochs if it doesn't exist (required for GPT configs)
    if 'mini_epochs' not in data['params']['config']:
        # Insert mini_epochs after max_epochs if it exists, otherwise add it to config
        if 'max_epochs' in data['params']['config']:
            # We'll add it after max_epochs when writing, but for now just set it
            data['params']['config']['mini_epochs'] = 5
        else:
            data['params']['config']['mini_epochs'] = 5

    # Write the new YAML file
    with open(output_file, 'w') as new_yamlfile:
        yaml.safe_dump(data, new_yamlfile)