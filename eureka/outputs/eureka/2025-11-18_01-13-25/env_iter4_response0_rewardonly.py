@torch.jit.script
def compute_reward(root_states: torch.Tensor, up_vec: torch.Tensor, actions: torch.Tensor, progress_buf: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract velocity from root states (indices 7:10 are linear velocity)
    velocity = root_states[:, 7:10]
    
    # Main reward: encourage forward speed (x-direction)
    forward_speed = velocity[:, 0]  # Assuming x-axis is forward direction
    
    # Stability penalty: penalize when not upright (up_vec z-component should be 1.0 when upright)
    up_deviation = torch.abs(up_vec[:, 2] - 1.0)
    
    # Action penalty for energy efficiency
    action_penalty = -torch.sum(actions * actions, dim=-1)
    
    # Small time penalty to encourage efficiency (negative reward proportional to time step)
    time_penalty = -0.001 * progress_buf.float()
    
    # Temperature parameters for proper balance
    speed_temp = 0.1          # Lower temperature to prevent excessive dominance
    stability_temp = 10.0     # Higher temperature to make stability penalty more sensitive
    action_temp = 0.01        # Small temperature for action regularization
    
    # Apply transformations
    speed_reward = torch.clamp(forward_speed, min=0.0)  # Only reward positive forward speed
    stability_penalty = -stability_temp * up_deviation   # Linear penalty scaled by higher temperature
    action_penalty_shaped = torch.exp(action_temp * action_penalty)
    
    # Combine rewards
    reward = speed_reward + stability_penalty + action_penalty_shaped + time_penalty
    
    reward_components = {
        "speed_reward": speed_reward,
        "stability_penalty": stability_penalty,
        "action_penalty": action_penalty_shaped,
        "time_penalty": time_penalty
    }
    
    return reward, reward_components
