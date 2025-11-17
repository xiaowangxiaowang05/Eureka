@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dof_vel: torch.Tensor, actions: torch.Tensor, contact_force_tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract relevant state information
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    
    # Compute forward direction (assuming target is in x-direction)
    to_target = targets - torso_position
    to_target[:, 2] = 0.0  # ignore vertical component
    target_distance = torch.norm(to_target, p=2, dim=-1)
    
    # Main goal reward: forward velocity (x-component of velocity in world frame)
    # Since we want to run as fast as possible, we maximize forward speed
    forward_vel = velocity[:, 0]  # assuming x-axis is forward direction
    
    # Alternative: if the environment uses a different forward direction,
    # we could use the heading vector, but based on observation code,
    # the target is used to compute heading, so x-axis appears to be forward
    
    # Stability penalties
    # 1. Penalize excessive joint velocities (energy efficiency)
    joint_vel_penalty = -torch.sum(torch.square(dof_vel), dim=-1)
    
    # 2. Penalize large actions (control effort)
    action_penalty = -torch.sum(torch.square(actions), dim=-1)
    
    # 3. Penalize contact forces (roughly indicates instability or falling)
    # contact_force_tensor shape: [num_envs, num_bodies, 3]
    # We'll sum over all contact forces
    contact_penalty = -torch.sum(torch.square(contact_force_tensor), dim=[1, 2])
    
    # 4. Penalize non-forward movement (lateral and vertical velocity)
    lateral_vel_penalty = -torch.square(velocity[:, 1])  # y-axis (sideways)
    vertical_vel_penalty = -torch.square(velocity[:, 2])  # z-axis (up/down)
    
    # Combine rewards with appropriate scaling
    velocity_reward = forward_vel
    stability_reward = (
        0.05 * joint_vel_penalty + 
        0.01 * action_penalty + 
        0.001 * contact_penalty + 
        0.1 * lateral_vel_penalty + 
        0.1 * vertical_vel_penalty
    )
    
    total_reward = velocity_reward + stability_reward
    
    reward_dict = {
        "velocity_reward": velocity_reward,
        "joint_vel_penalty": joint_vel_penalty,
        "action_penalty": action_penalty,
        "contact_penalty": contact_penalty,
        "lateral_vel_penalty": lateral_vel_penalty,
        "vertical_vel_penalty": vertical_vel_penalty,
        "stability_reward": stability_reward
    }
    
    return total_reward, reward_dict
