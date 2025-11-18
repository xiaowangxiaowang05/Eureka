@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, actions: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract relevant state information
    torso_position = root_states[:, 0:3]
    torso_rotation = root_states[:, 3:7]
    velocity = root_states[:, 7:10]
    ang_velocity = root_states[:, 10:13]
    
    # Calculate direction to target (horizontal plane only)
    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    target_distance = torch.norm(to_target, dim=-1)
    target_direction = to_target / (target_distance.unsqueeze(-1) + 1e-8)
    
    # Get forward direction of the humanoid (assuming forward is x-axis in local frame)
    # Convert quaternion to forward vector
    qw, qx, qy, qz = torso_rotation[:, 0], torso_rotation[:, 1], torso_rotation[:, 2], torso_rotation[:, 3]
    forward_x = 2 * (qw * qx + qy * qz)
    forward_y = 2 * (qw * qy - qx * qz)
    forward_z = 1 - 2 * (qx * qx + qy * qy)
    forward_vec = torch.stack([forward_x, forward_y, torch.zeros_like(forward_x)], dim=-1)
    forward_norm = torch.norm(forward_vec[:, :2], dim=-1)
    forward_dir = forward_vec.clone()
    forward_dir[:, :2] = forward_vec[:, :2] / (forward_norm.unsqueeze(-1) + 1e-8)
    
    # Component 1: Forward velocity reward (velocity in target direction)
    velocity_magnitude = torch.norm(velocity[:, :2], dim=-1)
    velocity_in_target_dir = torch.sum(velocity[:, :2] * target_direction[:, :2], dim=-1)
    forward_vel_reward_temp = 0.1
    forward_vel_reward = torch.exp(velocity_in_target_dir / forward_vel_reward_temp)
    
    # Component 2: Upright posture penalty (penalize low height)
    height = torso_position[:, 2]
    height_penalty_temp = 0.5
    height_penalty = torch.exp(-torch.abs(height - 0.85) / height_penalty_temp)
    
    # Component 3: Action smoothness penalty
    action_cost_temp = 0.1
    action_cost = torch.sum(actions * actions, dim=-1)
    action_smoothness_penalty = torch.exp(-action_cost / action_cost_temp)
    
    # Component 4: Orientation alignment with target
    orientation_alignment_temp = 0.5
    forward_alignment = torch.sum(forward_dir[:, :2] * target_direction[:, :2], dim=-1)
    orientation_reward = torch.exp(forward_alignment / orientation_alignment_temp)
    
    # Combine rewards with appropriate weights
    reward = (
        1.0 * forward_vel_reward +
        0.5 * height_penalty +
        0.1 * action_smoothness_penalty +
        0.3 * orientation_reward
    )
    
    reward_components = {
        "forward_vel_reward": forward_vel_reward,
        "height_penalty": height_penalty,
        "action_smoothness_penalty": action_smoothness_penalty,
        "orientation_reward": orientation_reward
    }
    
    return reward, reward_components
