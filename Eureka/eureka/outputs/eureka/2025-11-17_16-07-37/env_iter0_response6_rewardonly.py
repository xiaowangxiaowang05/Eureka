@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dof_vel: torch.Tensor, actions: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract relevant states
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    
    # Compute forward direction (assuming target is straight ahead in x-direction)
    to_target = targets - torso_position
    to_target[:, 2] = 0.0  # ignore vertical component
    target_dist = torch.norm(to_target, p=2, dim=-1)
    
    # Forward velocity reward (main goal: run fast)
    forward_vel = velocity[:, 0]  # assuming x-axis is forward direction
    forward_reward = forward_vel
    
    # Upright posture reward (stability)
    up_reward = up_vec[:, 2]  # z-component of up vector (should be close to 1 when upright)
    
    # Action regularization penalty (control effort)
    action_penalty = -torch.sum(actions ** 2, dim=-1)
    
    # Joint velocity penalty (smooth movement)
    joint_vel_penalty = -torch.sum(dof_vel ** 2, dim=-1)
    
    # Combine rewards with appropriate scaling
    total_reward = (
        forward_reward * 1.0 +
        up_reward * 0.5 +
        action_penalty * 0.01 +
        joint_vel_penalty * 0.0001
    )
    
    reward_dict = {
        "forward_reward": forward_reward,
        "up_reward": up_reward,
        "action_penalty": action_penalty,
        "joint_vel_penalty": joint_vel_penalty
    }
    
    return total_reward, reward_dict
