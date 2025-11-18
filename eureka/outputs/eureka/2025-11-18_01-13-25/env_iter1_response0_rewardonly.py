@torch.jit.script
def compute_reward(root_states: torch.Tensor, up_vec: torch.Tensor, actions: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract velocity from root states (indices 7:10 are linear velocity)
    velocity = root_states[:, 7:10]
    
    # Calculate horizontal speed (ignore vertical component for running speed)
    horizontal_speed = torch.sqrt(velocity[:, 0] * velocity[:, 0] + velocity[:, 1] * velocity[:, 1])
    
    # Main reward: encourage high horizontal speed (linear scaling for better gradient)
    speed_reward = horizontal_speed
    
    # Stability penalty: directly penalize deviation from upright posture
    # When upright, up_vec[:, 2] should be close to 1.0
    # Penalize how much it deviates from 1.0
    up_deviation = torch.abs(up_vec[:, 2] - 1.0)
    stability_penalty = -up_deviation * 10.0  # Scale penalty to be meaningful
    
    # Action penalty to encourage energy efficiency
    action_penalty = -torch.sum(actions * actions, dim=-1) * 0.1
    
    # Temperature parameters for reward shaping
    speed_temp = 0.1  # Lower temperature to prevent exponential explosion
    stability_temp = 1.0
    action_temp = 1.0
    
    # Apply transformations
    speed_reward_shaped = torch.exp(speed_temp * speed_reward)
    stability_reward_shaped = torch.exp(stability_temp * stability_penalty)
    action_penalty_shaped = torch.exp(action_temp * action_penalty)
    
    # Combine rewards
    reward = speed_reward_shaped + stability_reward_shaped + action_penalty_shaped
    
    reward_components = {
        "speed_reward": speed_reward_shaped,
        "stability_reward": stability_reward_shaped,
        "action_penalty": action_penalty_shaped
    }
    
    return reward, reward_components
