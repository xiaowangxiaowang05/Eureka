@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, object_angvel: torch.Tensor, actions: torch.Tensor, progress_buf: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute rotation difference quaternion
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    
    # Compute angle between current and goal orientation
    rot_error = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))
    
    # Main goal reward: encourage alignment with target orientation
    rot_reward_temp = 1.0
    rot_reward = torch.exp(-rot_error / rot_reward_temp)
    
    # Penalty for excessive angular velocity (stability)
    angvel_penalty_temp = 2.0
    angvel_penalty = torch.sum(torch.square(object_angvel), dim=-1)
    angvel_reward = torch.exp(-angvel_penalty / angvel_penalty_temp)
    
    # Action regularization penalty
    action_penalty_temp = 0.01
    action_penalty = torch.sum(torch.square(actions), dim=-1)
    action_reward = torch.exp(-action_penalty * action_penalty_temp)
    
    # Total reward as product of components
    total_reward = rot_reward * angvel_reward * action_reward
    
    reward_dict = {
        "rot_reward": rot_reward,
        "angvel_reward": angvel_reward,
        "action_reward": action_reward,
        "rot_error": rot_error,
        "angvel_penalty": angvel_penalty,
        "action_penalty": action_penalty
    }
    
    return total_reward, reward_dict
