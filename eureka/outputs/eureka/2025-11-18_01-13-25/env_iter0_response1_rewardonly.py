@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor, up_vec: torch.Tensor, heading_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract velocity components
    velocity = root_states[:, 7:10]  # linear velocity
    ang_velocity = root_states[:, 10:13]  # angular velocity
    
    # Main goal reward: forward velocity in heading direction
    # Project velocity onto heading direction to get forward speed
    forward_velocity = torch.sum(velocity * heading_vec, dim=-1)
    velocity_reward_temp = 0.1
    velocity_reward = torch.exp(forward_velocity / velocity_reward_temp)
    
    # Stability penalty: penalize deviation from upright posture
    # up_vec should be close to [0, 0, 1] when upright
    target_up = torch.zeros_like(up_vec)
    target_up[:, 2] = 1.0
    up_deviation = torch.norm(up_vec - target_up, dim=-1)
    stability_temp = 0.5
    stability_reward = torch.exp(-up_deviation / stability_temp)
    
    # Angular velocity penalty: penalize excessive spinning/rolling
    ang_vel_magnitude = torch.norm(ang_velocity, dim=-1)
    ang_vel_temp = 0.2
    ang_vel_penalty = torch.exp(-ang_vel_magnitude / ang_vel_temp)
    
    # Action penalty: discourage large actions for energy efficiency
    action_magnitude = torch.norm(actions, dim=-1)
    action_temp = 0.1
    action_penalty = torch.exp(-action_magnitude / action_temp)
    
    # Combine rewards with appropriate weights
    total_reward = velocity_reward * stability_reward * ang_vel_penalty * action_penalty
    
    reward_components = {
        "velocity_reward": velocity_reward,
        "stability_reward": stability_reward,
        "ang_vel_penalty": ang_vel_penalty,
        "action_penalty": action_penalty
    }
    
    return total_reward, reward_components
