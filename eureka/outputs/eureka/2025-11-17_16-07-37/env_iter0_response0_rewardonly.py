@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dof_vel: torch.Tensor, actions: torch.Tensor, contact_force_tensor: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract linear velocity in world frame
    velocity = root_states[:, 7:10]
    
    # Compute forward direction (assuming target is along +x initially)
    to_target = targets - root_states[:, 0:3]
    to_target[:, 2] = 0.0  # ignore vertical component
    target_dist = torch.norm(to_target, p=2, dim=-1)
    target_dir = to_target / (target_dist.unsqueeze(-1) + 1e-8)
    
    # Project velocity onto target direction to get forward speed
    forward_speed = torch.sum(velocity * target_dir, dim=-1)
    
    # Main reward: encourage high forward speed
    speed_reward = forward_speed
    
    # Penalty for excessive action magnitude (energy efficiency)
    action_penalty = -torch.sum(actions ** 2, dim=-1) * 0.01
    
    # Penalty for high joint velocities (smoothness)
    joint_vel_penalty = -torch.sum(dof_vel ** 2, dim=-1) * 0.0001
    
    # Penalty for large contact forces (roughness of movement)
    contact_penalty = -torch.sum(contact_force_tensor ** 2, dim=-1) * 0.00001
    
    # Combine rewards
    total_reward = speed_reward + action_penalty + joint_vel_penalty + contact_penalty
    
    reward_dict = {
        "speed_reward": speed_reward,
        "action_penalty": action_penalty,
        "joint_vel_penalty": joint_vel_penalty,
        "contact_penalty": contact_penalty
    }
    
    return total_reward, reward_dict
