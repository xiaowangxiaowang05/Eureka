@torch.jit.script
def compute_reward(root_states: torch.Tensor, up_vec: torch.Tensor, actions: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract velocity from root states (indices 7:10 are linear velocity)
    velocity = root_states[:, 7:10]
    
    # Main reward: encourage forward speed (x-direction)
    forward_speed = velocity[:, 0]  # Assuming x-axis is forward direction
    
    # Stability penalty: penalize when not upright (up_vec z-component should be 1.0 when upright)
    up_deviation = torch.abs(up_vec[:, 2] - 1.0)
    
    # Action penalty for energy efficiency
    action_penalty = -torch.sum(actions * actions, dim=-1)
    
    # Temperature parameters for proper balance
    speed_temp = 0.5        # Higher temperature to make speed reward more impactful
    stability_temp = 0.1    # Low temperature to keep stability penalty reasonable
    action_temp = 0.01      # Small temperature for action regularization
    
    # Apply transformations
    speed_reward = torch.exp(speed_temp * torch.clamp(forward_speed, min=0.0))  # Only reward positive forward speed
    stability_penalty = -torch.exp(stability_temp * up_deviation)  # Negative penalty for instability
    action_penalty_shaped = torch.exp(action_temp * action_penalty)
    
    # Combine rewards
    reward = speed_reward + stability_penalty + action_penalty_shaped
    
    reward_components = {
        "speed_reward": speed_reward,
        "stability_penalty": stability_penalty,
        "action_penalty": action_penalty_shaped
    }
    
    return reward, reward_components
