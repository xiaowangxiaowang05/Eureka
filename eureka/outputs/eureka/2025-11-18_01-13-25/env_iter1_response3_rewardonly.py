@torch.jit.script
def compute_reward(root_states: torch.Tensor, up_vec: torch.Tensor, actions: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract velocity from root states (indices 7:10 are linear velocity)
    velocity = root_states[:, 7:10]
    
    # Calculate horizontal speed (ignore vertical component for running speed)
    horizontal_speed = torch.sqrt(velocity[:, 0] * velocity[:, 0] + velocity[:, 1] * velocity[:, 1])
    
    # Main reward: encourage high horizontal speed
    speed_reward = horizontal_speed
    
    # Improved stability reward: use dot product with vertical axis [0, 0, 1]
    # up_vec[:, 2] is the z-component, which equals the dot product with [0, 0, 1]
    # This gives 1.0 when perfectly upright, 0.0 when horizontal, negative when inverted
    stability_reward = up_vec[:, 2]
    
    # Action penalty to encourage energy efficiency
    action_penalty = -torch.sum(actions * actions, dim=-1)
    
    # Temperature parameters for reward shaping - reduced speed_temp to prevent domination
    speed_temp = 0.1  # Significantly reduced from 1.0 to prevent extreme values
    stability_temp = 2.0  # Increased to give more importance to stability
    action_temp = 0.01
    
    # Apply exponential transformations with temperatures
    speed_reward_shaped = torch.exp(speed_temp * speed_reward)
    stability_reward_shaped = torch.exp(stability_temp * stability_reward)
    action_penalty_shaped = torch.exp(action_temp * action_penalty)
    
    # Combine rewards
    reward = speed_reward_shaped + stability_reward_shaped + action_penalty_shaped
    
    reward_components = {
        "speed_reward": speed_reward_shaped,
        "stability_reward": stability_reward_shaped,
        "action_penalty": action_penalty_shaped
    }
    
    return reward, reward_components
