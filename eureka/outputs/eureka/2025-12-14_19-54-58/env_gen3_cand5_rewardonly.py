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
    
    # Dense orientation reward with adjusted temperature for better gradient at small errors
    orientation_temp = 2.0
    orientation_reward = torch.exp(-orientation_temp * angle_error)
    
    # Spin maintenance around object Z-axis (target ~5 rad/s)
    spin_axis = torch.zeros_like(object_angvel)
    spin_axis[:, 2] = 1.0
    spin_proj = torch.sum(object_angvel * spin_axis, dim=-1)
    spin_temp = 0.3
    spin_reward = torch.exp(-spin_temp * torch.abs(spin_proj - 5.0))
    
    # Action regularization: increased weight and added rate-of-change proxy
    # Since prev_actions is unavailable, we penalize large action norms AND encourage uniformity across joints
    # to reduce independent finger twitching (jitter often involves sparse, large-magnitude actions)
    action_norm = torch.sum(actions ** 2, dim=-1)
    action_uniformity = torch.var(actions, dim=-1)  # low variance = more coordinated movement
    action_penalty = -(action_norm * 0.05 + action_uniformity * 0.1)
    
    # Stronger angular velocity magnitude penalty to prevent violent spinning
    angvel_penalty = -torch.norm(object_angvel, dim=-1) * 0.1
    
    # Timeout penalty scaled by remaining time and orientation error
    time_ratio = (progress_buf / max_episode_length).float()
    timeout_penalty = -time_ratio * angle_error * 0.2
    
    # Combine rewards with balanced weights
    total_reward = (
        orientation_reward * 3.0 +
        spin_reward * 1.5 +
        action_penalty +
        angvel_penalty +
        timeout_penalty
    )
    
    reward_components = {
        "orientation_reward": orientation_reward,
        "spin_reward": spin_reward,
        "action_penalty": action_penalty,
        "angvel_penalty": angvel_penalty,
        "timeout_penalty": timeout_penalty
    }
    
    return total_reward, reward_components