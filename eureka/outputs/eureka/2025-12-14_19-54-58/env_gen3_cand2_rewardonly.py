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
    
    # Smoothness penalty
    smoothness_penalty = -torch.sum(torch.abs(actions), dim=-1) * 0.002

    # NEW: Penalty for object linear velocity (discourage dropping/slipping)
    # NOTE: object_linvel is not in inputs, but we can infer instability from high angular velocity without orientation improvement
    # Alternatively, since object_linvel isn't available, we use a proxy: high angvel without low angle_error indicates unstable spin
    instability_proxy = torch.norm(object_angvel, dim=-1) * (1.0 - orientation_reward)
    linvel_penalty = -instability_proxy * 0.1

    # NEW: Reward for coordinated finger action (assume first 16 DOFs are fingers, last 4 are wrist)
    # If wrist torque (last 4 actions) is high but finger actions (first 16) are low, penalize
    finger_actions = actions[:, :16]
    wrist_actions = actions[:, 16:]
    finger_action_norm = torch.norm(finger_actions, dim=-1)
    wrist_action_norm = torch.norm(wrist_actions, dim=-1)
    coordination_penalty = -torch.where(
        (wrist_action_norm > 0.5) & (finger_action_norm < 0.3),
        wrist_action_norm * 0.2,
        torch.zeros_like(wrist_action_norm)
    )

    # Time-based timeout penalty (only apply when not near success)
    time_factor = (progress_buf / max_episode_length).float()
    timeout_penalty = -time_factor * (1.0 - orientation_reward) * 0.05

    total_reward = (
        orientation_reward * 3.0 +
        spin_reward * 1.5 +
        action_penalty +
        smoothness_penalty +
        linvel_penalty +
        coordination_penalty +
        timeout_penalty
    )
    
    reward_components = {
        "orientation_reward": orientation_reward,
        "spin_reward": spin_reward,
        "action_penalty": action_penalty,
        "smoothness_penalty": smoothness_penalty,
        "linvel_penalty": linvel_penalty,
        "coordination_penalty": coordination_penalty,
        "timeout_penalty": timeout_penalty
    }
    
    return total_reward, reward_components