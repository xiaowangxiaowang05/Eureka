@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor, up_vec: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract relevant state information
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    
    # Forward direction is along the heading (we want to maximize forward speed)
    # Since we don't have explicit heading, we'll use the global x-direction as forward
    # This assumes the humanoid should run in the positive x direction
    forward_velocity = velocity[:, 0]  # x-component of velocity
    
    # Torso height (z-coordinate)
    torso_height = torso_position[:, 2]
    
    # Upright posture: up_vec[:, 2] should be close to 1.0 when upright
    upright_posture = up_vec[:, 2]
    
    # Action regularization (L2 norm of actions)
    action_norm = torch.sum(actions ** 2, dim=-1)
    
    # Reward components
    # Main goal: maximize forward velocity
    forward_velocity_temp = 2.0
    forward_reward = torch.exp(forward_velocity_temp * forward_velocity)
    
    # Stability: encourage upright posture
    upright_temp = 5.0
    upright_reward = torch.exp(upright_temp * (upright_posture - 1.0))
    
    # Action regularization to encourage energy efficiency
    action_temp = 0.01
    action_penalty = -action_temp * action_norm
    
    # Fall penalty: strong penalty if torso height is too low
    height_threshold = 0.8
    fall_penalty = torch.where(torso_height < height_threshold, 
                              torch.full_like(torso_height, -10.0), 
                              torch.zeros_like(torso_height))
    
    # Total reward
    total_reward = forward_reward + upright_reward + action_penalty + fall_penalty
    
    reward_components = {
        "forward_reward": forward_reward,
        "upright_reward": upright_reward,
        "action_penalty": action_penalty,
        "fall_penalty": fall_penalty
    }
    
    return total_reward, reward_components
