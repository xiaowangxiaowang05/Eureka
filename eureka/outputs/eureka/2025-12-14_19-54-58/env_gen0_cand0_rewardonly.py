@torch.jit.script
def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    shadow_hand_dof_vel: torch.Tensor,
    progress_buf: torch.Tensor,
    max_episode_length: float
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error using quaternion distance
    # quat_mul(q1, quat_conjugate(q2)) gives relative rotation from q2 to q1
    rel_quat = torch.mul(object_rot, torch.cat([
        -goal_rot[:, 0:1],
        goal_rot[:, 1:2],
        goal_rot[:, 2:3],
        goal_rot[:, 3:4]
    ], dim=-1))  # Manual conjugate since quat_conjugate may not be available
    
    # Normalize the relative quaternion
    rel_quat = rel_quat / (torch.norm(rel_quat, dim=-1, keepdim=True) + 1e-8)
    
    # Orientation error: angle of rotation (0 to pi)
    orientation_error = 2.0 * torch.asin(torch.clamp(torch.abs(rel_quat[:, 0]), min=0.0, max=1.0))
    
    # Temperature for orientation reward shaping
    ori_temp = 1.0
    orientation_reward = torch.exp(-ori_temp * orientation_error)
    
    # Angular velocity alignment: encourage spinning around correct axis
    # Target angular velocity direction should align with the axis of rotation needed
    # For simplicity, we assume any spin that reduces orientation error is good
    # But we penalize excessive angular velocity that might destabilize
    angvel_norm = torch.norm(object_angvel, dim=-1)
    angvel_temp = 0.25
    angvel_penalty = torch.exp(-angvel_temp * angvel_norm)
    
    # Action regularization: penalize large actions
    action_penalty = torch.sum(actions ** 2, dim=-1)
    action_reg = -0.01 * action_penalty
    
    # Joint velocity regularization: prevent jitter
    joint_vel_penalty = torch.sum(shadow_hand_dof_vel ** 2, dim=-1)
    joint_vel_reg = -0.001 * joint_vel_penalty
    
    # Success bonus: when orientation error is very small
    success_threshold = 0.1
    success_bonus = torch.where(orientation_error < success_threshold, 2.0, 0.0)
    
    # Time-based survival reward to encourage longer episodes
    survival_reward = 0.01 * (progress_buf / max_episode_length)
    
    # Combine rewards
    total_reward = (
        orientation_reward +
        0.5 * angvel_penalty +
        action_reg +
        joint_vel_reg +
        success_bonus +
        survival_reward
    )
    
    reward_components = {
        "orientation_reward": orientation_reward,
        "angvel_penalty": angvel_penalty,
        "action_reg": action_reg,
        "joint_vel_reg": joint_vel_reg,
        "success_bonus": success_bonus,
        "survival_reward": survival_reward
    }
    
    return total_reward, reward_components