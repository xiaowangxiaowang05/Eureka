@torch.jit.script
def compute_reward(root_states: torch.Tensor, up_vec: torch.Tensor, actions: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract velocity from root states (indices 7:10 are linear velocity)
    velocity = root_states[:, 7:10]
    
    # Calculate forward speed (x-component for running forward)
    # Assuming the humanoid runs along the x-axis
    forward_speed = velocity[:, 0]
    
    # Main reward: encourage high forward speed
    # Clip to prevent extremely negative rewards when moving backward
    speed_reward = torch.clamp(forward_speed, min=0.0)
    
    # Stability penalty: penalize when not upright
    # up_vec z-component should be 1.0 when perfectly upright
    up_deviation = torch.abs(up_vec[:, 2] - 1.0)
    stability_penalty = -up_deviation
    
    # Action penalty to encourage energy efficiency
    action_penalty = -torch.sum(actions * actions, dim=-1)
    
    # Weight parameters to balance components appropriately
    speed_weight = 2.0      # High weight to encourage running
    stability_weight = 1.0  # Moderate weight for stability
    action_weight = 0.01    # Small weight for action regularization
    
    # Combine rewards with appropriate weights
    reward = (speed_weight * speed_reward + 
              stability_weight * stability_penalty + 
              action_weight * action_penalty)
    
    reward_components = {
        "speed_reward": speed_weight * speed_reward,
        "stability_penalty": stability_weight * stability_penalty,
        "action_penalty": action_weight * action_penalty
    }
    
    return reward, reward_components
