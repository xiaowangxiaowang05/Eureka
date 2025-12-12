@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, object_angvel: torch.Tensor, actions: torch.Tensor, object_linvel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error using quaternion difference
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    angle_error = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))
    
    # Main goal reward: encourage alignment with target orientation
    orientation_reward_temp = 10.0
    orientation_reward = torch.exp(-orientation_reward_temp * angle_error)
    
    # Penalize excessive angular velocity (for stable spinning)
    angvel_penalty_temp = 0.1
    angvel_penalty = torch.exp(-angvel_penalty_temp * torch.norm(object_angvel, p=2, dim=-1))
    
    # Penalize large linear velocity (keep object centered)
    linvel_penalty_temp = 2.0
    linvel_penalty = torch.exp(-linvel_penalty_temp * torch.norm(object_linvel, p=2, dim=-1))
    
    # Action regularization to encourage energy efficiency
    action_penalty_temp = 0.01
    action_penalty = torch.exp(-action_penalty_temp * torch.norm(actions, p=2, dim=-1))
    
    # Combine rewards
    total_reward = orientation_reward * angvel_penalty * linvel_penalty * action_penalty
    
    reward_dict = {
        "orientation_reward": orientation_reward,
        "angvel_penalty": angvel_penalty,
        "linvel_penalty": linvel_penalty,
        "action_penalty": action_penalty
    }
    
    return total_reward, reward_dict
