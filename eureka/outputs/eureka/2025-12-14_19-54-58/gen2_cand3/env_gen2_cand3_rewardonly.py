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
    
    orientation_temp = 2.0
    orientation_reward = torch.exp(-orientation_temp * angle_error)
    
    # Spin maintenance around object Z-axis (target ~5 rad/s)
    spin_axis = torch.zeros_like(object_angvel)
    spin_axis[:, 2] = 1.0
    spin_proj = torch.sum(object_angvel * spin_axis, dim=-1)
    spin_temp = 0.3
    spin_reward = torch.exp(-spin_temp * torch.abs(spin_proj - 5.0))
    
    # Action regularization (L2)
    action_penalty = -torch.sum(actions ** 2, dim=-1) * 0.005
    
    # Smoothness penalty (approximate jerk/acceleration via action magnitude; no history available)
    smoothness_penalty = -torch.sum(torch.abs(actions), dim=-1) * 0.002
    
    # Default pose deviation penalty (weaker than before to allow necessary motion)
    default_pose_penalty = -torch.sum(actions ** 2, dim=-1) * 0.008

    # Time-based timeout penalty (only apply when not near success)
    time_factor = (progress_buf / max_episode_length).float()
    timeout_penalty = -time_factor * (1.0 - orientation_reward) * 0.05

    total_reward = (
        orientation_reward * 3.0 +
        spin_reward * 1.5 +
        action_penalty +
        smoothness_penalty +
        default_pose_penalty +
        timeout_penalty
    )
    
    reward_components = {
        "orientation_reward": orientation_reward,
        "spin_reward": spin_reward,
        "action_penalty": action_penalty,
        "smoothness_penalty": smoothness_penalty,
        "default_pose_penalty": default_pose_penalty,
        "timeout_penalty": timeout_penalty
    }
    
    return total_reward, reward_components