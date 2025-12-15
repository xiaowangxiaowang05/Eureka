@torch.jit.script
def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    progress_buf: torch.Tensor,
    max_episode_length: float
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Orientation error (dense reward)
    rel_quat = quat_mul(object_rot, quat_conjugate(goal_rot))
    angle_error = 2.0 * torch.asin(torch.clamp(torch.norm(rel_quat[:, 0:3], dim=-1), min=0.0, max=1.0))
    orientation_temp = 2.0
    orientation_reward = torch.exp(-orientation_temp * angle_error)
    
    # Spin consistency: penalize uneven rotation (reduce wobble/jitter in spin axis)
    angvel_mag = torch.norm(object_angvel, dim=-1)
    angvel_var = torch.var(object_angvel, dim=-1)
    spin_consistency_temp = 5.0
    spin_consistency_reward = torch.exp(-spin_consistency_temp * angvel_var)
    
    # Action smoothness (proxy for acceleration/jitter): stronger penalty on large actions
    action_penalty = -torch.sum(actions ** 2, dim=-1) * 0.1
    
    # Angular velocity magnitude penalty to avoid violent spinning
    angvel_penalty = -angvel_mag * 0.1
    
    # Timeout penalty (scaled by remaining time and orientation error)
    time_ratio = (progress_buf / max_episode_length).float()
    timeout_penalty = -time_ratio * (1.0 - orientation_reward) * 0.05
    
    # Total reward with balanced weights
    total_reward = (
        orientation_reward * 3.0 +
        spin_consistency_reward * 1.0 +
        action_penalty +
        angvel_penalty +
        timeout_penalty
    )
    
    reward_components = {
        "orientation_reward": orientation_reward,
        "spin_consistency_reward": spin_consistency_reward,
        "action_penalty": action_penalty,
        "angvel_penalty": angvel_penalty,
        "timeout_penalty": timeout_penalty
    }
    
    return total_reward, reward_components