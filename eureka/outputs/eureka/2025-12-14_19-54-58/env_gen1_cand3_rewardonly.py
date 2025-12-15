@torch.jit.script
def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    progress_buf: torch.Tensor,
    max_episode_length: float
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error using quaternion distance
    rel_quat = quat_mul(object_rot, quat_conjugate(goal_rot))
    angle_error = 2.0 * torch.asin(torch.clamp(torch.norm(rel_quat[:, 0:3], dim=-1), min=0.0, max=1.0))
    
    orientation_temp = 1.0
    orientation_reward = torch.exp(-orientation_temp * angle_error)
    
    # Spin maintenance around object Z-axis
    spin_axis = torch.zeros_like(object_angvel)
    spin_axis[:, 2] = 1.0
    spin_proj = torch.sum(object_angvel * spin_axis, dim=-1)
    spin_temp = 0.5
    spin_reward = torch.exp(-spin_temp * torch.abs(spin_proj - 5.0))
    
    # Action regularization
    action_penalty = -torch.sum(actions ** 2, dim=-1) * 0.01
    
    # Time-based timeout penalty
    time_factor = (progress_buf / max_episode_length).float()
    timeout_penalty = -time_factor * (1.0 - orientation_reward) * 0.1
    
    # NEW: Smoothness penalty (difference between consecutive actions is not available here,
    # so we approximate by penalizing high-frequency changes indirectly via action derivative proxy;
    # however, since prev_actions isn't available, we instead penalize large action magnitudes more strongly
    # and add a default pose deviation penalty below)
    
    # NEW: Deviation from default joint positions (assume default is zero for all dofs as per env setup)
    # Since shadow_hand_default_dof_pos is zero, we can use actions as proxy for joint position deviation
    # (because actions map to target positions relative to default in non-relative control)
    default_pose_penalty = -torch.sum(actions ** 2, dim=-1) * 0.02  # stronger than action penalty alone

    # NEW: Joint acceleration proxy – since we don't have dof_acc, we use action jerk approximation
    # But without history, we cannot compute real acceleration; instead, we rely on default_pose_penalty 
    # and action_penalty to discourage extreme movements. Add small torque-like penalty via action magnitude.
    torque_penalty = -torch.sum(torch.abs(actions), dim=-1) * 0.005
    
    total_reward = (
        orientation_reward * 2.0 +
        spin_reward * 1.0 +
        action_penalty +
        timeout_penalty +
        default_pose_penalty +
        torque_penalty
    )
    
    reward_components = {
        "orientation_reward": orientation_reward,
        "spin_reward": spin_reward,
        "action_penalty": action_penalty,
        "timeout_penalty": timeout_penalty,
        "default_pose_penalty": default_pose_penalty,
        "torque_penalty": torque_penalty
    }
    
    return total_reward, reward_components