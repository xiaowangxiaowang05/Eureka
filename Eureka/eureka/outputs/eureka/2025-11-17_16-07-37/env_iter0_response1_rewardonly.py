@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dof_vel: torch.Tensor, actions: torch.Tensor, contact_force_tensor: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract linear velocity in world frame
    velocity = root_states[:, 7:10]
    
    # Compute forward direction (assuming y-axis is forward based on typical humanoid setups)
    # If the environment uses a different forward axis, this may need adjustment
    # Based on observation code, heading is computed from torso_rotation and basis vectors
    # For running speed reward, we focus on velocity magnitude in the horizontal plane
    
    # Horizontal speed (ignore vertical component)
    horizontal_velocity = velocity.clone()
    horizontal_velocity[:, 2] = 0.0
    speed = torch.norm(horizontal_velocity, p=2, dim=-1)
    
    # Reward for moving fast forward
    run_reward = speed
    
    # Action regularization penalty
    action_penalty = -torch.sum(actions ** 2, dim=-1) * 0.01
    
    # Contact force penalty to discourage excessive ground impact
    contact_penalty = -torch.sum(contact_force_tensor ** 2, dim=-1) * 0.0001
    
    # Total reward
    total_reward = run_reward + action_penalty + contact_penalty
    
    reward_dict = {
        "run_reward": run_reward,
        "action_penalty": action_penalty,
        "contact_penalty": contact_penalty
    }
    
    return total_reward, reward_dict
