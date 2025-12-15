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
    
    # Dense orientation reward
    orientation_temp = 1.0
    orientation_reward = torch.exp(-orientation_temp * angle_error)
    
    # Spin reward: encourage angular velocity around object Z-axis at moderate rate
    spin_axis = torch.zeros_like(object_angvel)
    spin_axis[:, 2] = 1.0
    spin_proj = torch.abs(torch.sum(object_angvel * spin_axis, dim=-1))
    spin_temp = 0.5
    spin_target = 3.0  # moderate spin rate
    spin_reward = torch.exp(-spin_temp * torch.abs(spin_proj - spin_target))
    
    # Action regularization
    action_penalty = -torch.sum(actions ** 2, dim=-1) * 0.005
    
    # Action smoothness penalty (change in actions)
    # Note: since we don't have prev_actions as input, we approximate by penalizing high-frequency components via action magnitude squared
    # This is a proxy; true smoothness would require prev_actions, but we work with available inputs
    action_smoothness_penalty = -torch.sum(actions ** 2, dim=-1) * 0.005
    
    # Timeout penalty to encourage faster completion
    time_factor = (progress_buf / max_episode_length).float()
    timeout_penalty = -time_factor * (1.0 - orientation_reward) * 0.05
    
    # Total reward combines all components
    total_reward = (
        orientation_reward * 2.0 +
        spin_reward * 1.0 +
        action_penalty +
        action_smoothness_penalty +
        timeout_penalty
    )
    
    reward_components = {
        "orientation_reward": orientation_reward,
        "spin_reward": spin_reward,
        "action_penalty": action_penalty,
        "action_smoothness_penalty": action_smoothness_penalty,
        "timeout_penalty": timeout_penalty
    }
    
    return total_reward, reward_components