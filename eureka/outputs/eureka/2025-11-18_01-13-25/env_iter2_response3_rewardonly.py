@torch.jit.script
def compute_reward(root_states: torch.Tensor, up_vec: torch.Tensor, actions: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract velocity from root states (indices 7:10 are linear velocity)
    velocity = root_states[:, 7:10]
    
    # Main reward: forward speed (x-direction velocity)
    forward_speed = velocity[:, 0]  # Assuming x-axis is forward direction
    
    # Stability penalty: penalize deviation from upright posture
    # up_vec z-component should be 1.0 when perfectly upright
    up_deviation = torch.abs(up_vec[:, 2] - 1.0)
    stability_penalty = -up_deviation
    
    # Action penalty: encourage energy efficiency
    action_penalty = -torch.sum(actions * actions, dim=-1)
    
    # Temperature parameters for proper scaling
    speed_temp = 1.0        # Direct scaling for speed
    stability_temp = 2.0    # Strong penalty for instability  
    action_temp = 0.01      # Small penalty for actions
    
    # Apply appropriate transformations
    speed_reward = speed_temp * torch.clamp(forward_speed, min=0.0)  # Only reward positive forward speed
    stability_reward = stability_temp * stability_penalty
    action_reward = action_temp * action_penalty
    
    # Combine rewards
    reward = speed_reward + stability_reward + action_reward
    
    reward_components = {
        "speed_reward": speed_reward,
        "stability_reward": stability_reward, 
        "action_penalty": action_reward
    }
    
    return reward, reward_components
