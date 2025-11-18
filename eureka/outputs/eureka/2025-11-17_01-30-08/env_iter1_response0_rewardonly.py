@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, object_angvel: torch.Tensor, actions: torch.Tensor, object_linvel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error using quaternion difference
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    rot_error = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))
    
    # Main goal reward: encourage alignment with target orientation
    rot_reward_temp = 5.0
    rot_reward = torch.exp(-rot_error * rot_reward_temp)
    
    # Penalize excessive angular velocity (for stability)
    angvel_penalty_temp = 0.1
    angvel_penalty = torch.sum(torch.square(object_angvel), dim=-1)
    angvel_reward = torch.exp(-angvel_penalty * angvel_penalty_temp)
    
    # Penalize large actions (for smooth control)
    action_penalty_temp = 0.01
    action_penalty = torch.sum(torch.square(actions), dim=-1)
    action_reward = torch.exp(-action_penalty * action_penalty_temp)
    
    # Penalize linear movement of the object (keep it centered while spinning)
    linvel_penalty_temp = 1.0
    linvel_penalty = torch.sum(torch.square(object_linvel), dim=-1)
    linvel_reward = torch.exp(-linvel_penalty * linvel_penalty_temp)
    
    # Combine rewards
    total_reward = rot_reward * angvel_reward * action_reward * linvel_reward
    
    reward_dict = {
        "rot_reward": rot_reward,
        "angvel_reward": angvel_reward,
        "action_reward": action_reward,
        "linvel_reward": linvel_reward
    }
    
    return total_reward, reward_dict
