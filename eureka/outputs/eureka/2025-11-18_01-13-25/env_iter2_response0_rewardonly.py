@torch.jit.script
def compute_reward(root_states: torch.Tensor, up_vec: torch.Tensor, actions: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract velocity from root states (indices 7:10 are linear velocity)
    velocity = root_states[:, 7:10]
    
    # Main reward: encourage high horizontal speed (forward direction)
    horizontal_speed = torch.sqrt(velocity[:, 0] * velocity[:, 0] + velocity[:, 1] * velocity[:, 1])
    
    # Stability penalty: penalize when not upright (up_vec z-component should be close to 1)
    up_deviation = torch.abs(up_vec[:, 2] - 1.0)
    stability_penalty = -up_deviation
    
    # Action penalty to encourage energy efficiency
    action_penalty = -torch.sum(actions * actions, dim=-1)
    
    # Temperature parameters for reward shaping
    speed_temp = 2.0        # Higher temperature to emphasize speed
    stability_temp = 5.0    # High temperature for stability penalty to be significant when falling
    action_temp = 0.01      # Small temperature for action penalty
    
    # Apply transformations
    speed_reward_shaped = torch.exp(speed_temp * horizontal_speed)
    stability_penalty_shaped = torch.exp(stability_temp * stability_penalty)  # This will be small when fallen
    action_penalty_shaped = torch.exp(action_temp * action_penalty)
    
    # Combine rewards
    reward = speed_reward_shaped + stability_penalty_shaped + action_penalty_shaped
    
    reward_components = {
        "speed_reward": speed_reward_shaped,
        "stability_penalty": stability_penalty_shaped,
        "action_penalty": action_penalty_shaped
    }
    
    return reward, reward_components
