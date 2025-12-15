@torch.jit.script
def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    shadow_hand_dof_vel: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error using quaternion distance
    # quat_mul(q1, quat_conjugate(q2)) gives relative rotation from q2 to q1
    rel_quat = quat_mul(object_rot, quat_conjugate(goal_rot))
    # The w component of the relative quaternion relates to the angle difference
    # angle = 2 * acos(|w|), so we can use |w| as a proxy for alignment
    orientation_align = torch.abs(rel_quat[:, 3])  # w component
    
    # Temperature for orientation reward shaping
    orientation_temp = 1.0
    orientation_reward = torch.exp(-orientation_temp * (1.0 - orientation_align))
    
    # Angular velocity regularization: penalize excessive spinning once aligned
    # But allow some rotation to maintain stability during spin task
    angvel_norm = torch.norm(object_angvel, dim=-1)
    angvel_temp = 0.1
    angvel_penalty = torch.exp(-angvel_temp * angvel_norm)
    
    # Action regularization to encourage energy efficiency
    action_penalty = torch.sum(actions ** 2, dim=-1)
    action_reg_weight = 0.01
    action_reward = -action_reg_weight * action_penalty
    
    # Joint velocity regularization to prevent jitter
    joint_vel_norm = torch.sum(shadow_hand_dof_vel ** 2, dim=-1)
    joint_vel_weight = 0.001
    joint_vel_reward = -joint_vel_weight * joint_vel_norm
    
    # Combine rewards
    total_reward = (
        orientation_reward +
        0.5 * angvel_penalty +
        action_reward +
        joint_vel_reward
    )
    
    return total_reward, {
        "orientation_reward": orientation_reward,
        "angvel_penalty": angvel_penalty,
        "action_reward": action_reward,
        "joint_vel_reward": joint_vel_reward
    }