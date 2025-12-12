@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dof_vel: torch.Tensor, actions: torch.Tensor, contact_force_tensor: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract relevant states
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    
    # Compute forward direction (assuming y-axis is forward based on typical humanoid setups)
    # In the observation code, they zero out z-component of to_target and compute heading in x-y plane
    # So forward speed is velocity in the direction of current heading toward target
    to_target = targets - torso_position
    to_target[:, 2] = 0.0  # ignore height difference
    target_dist = torch.norm(to_target, p=2, dim=-1)
    
    # Avoid division by zero
    target_dir = to_target / (target_dist.unsqueeze(-1) + 1e-8)
    
    # Forward velocity component (projection of velocity onto target direction)
    vel_forward = torch.sum(velocity * target_dir, dim=-1)
    
    # Main goal reward: encourage high forward speed
    speed_reward = vel_forward
    
    # Stability penalties
    # Penalize large actions (energy efficiency)
    action_penalty = -torch.sum(actions ** 2, dim=-1)
    
    # Penalize contact forces (to discourage slamming into ground/objects)
    contact_penalty = -torch.sum(contact_force_tensor ** 2, dim=-1)
    
    # Penalize non-forward movement (sideways and vertical velocity)
    vel_lateral = velocity[:, 0]  # x-axis (sideways)
    vel_vertical = velocity[:, 2]  # z-axis (vertical)
    velocity_penalty = -torch.abs(vel_lateral) - torch.abs(vel_vertical)
    
    # Combine rewards with appropriate scaling
    total_reward = (
        speed_reward * 1.0 +
        action_penalty * 0.01 +
        contact_penalty * 0.001 +
        velocity_penalty * 0.5
    )
    
    reward_dict = {
        "speed_reward": speed_reward,
        "action_penalty": action_penalty,
        "contact_penalty": contact_penalty,
        "velocity_penalty": velocity_penalty
    }
    
    return total_reward, reward_dict
