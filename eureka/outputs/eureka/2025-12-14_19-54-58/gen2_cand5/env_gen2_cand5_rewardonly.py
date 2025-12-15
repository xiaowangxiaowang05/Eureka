@torch.jit.script
def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    progress_buf: torch.Tensor,
    max_episode_length: float
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Orientation error using quaternion distance
    rel_quat = quat_mul(object_rot, quat_conjugate(goal_rot))
    angle_error = 2.0 * torch.asin(torch.clamp(torch.norm(rel_quat[:, 0:3], dim=-1), min=0.0, max=1.0))
    
    # Dense orientation reward with appropriate temperature
    orientation_temp = 2.0
    orientation_reward = torch.exp(-orientation_temp * angle_error)
    
    # Spin maintenance around object Z-axis (target ~4-5 rad/s is reasonable; use Gaussian-like peak)
    spin_axis = torch.zeros_like(object_angvel)
    spin_axis[:, 2] = 1.0
    spin_proj = torch.sum(object_angvel * spin_axis, dim=-1)
    spin_target = 4.5
    spin_temp = 0.3
    spin_reward = torch.exp(-spin_temp * (spin_proj - spin_target)**2)
    
    # Action regularization: stronger penalty on magnitude to reduce jitter
    action_penalty = -torch.sum(actions ** 2, dim=-1) * 0.05
    
    # Penalty on total angular velocity magnitude to prevent violent spinning
    angvel_magnitude = torch.norm(object_angvel, dim=-1)
    angvel_penalty = -angvel_magnitude * 0.1
    
    # Time penalty scaled by remaining orientation error to encourage faster convergence
    time_ratio = (progress_buf / max_episode_length).float()
    timeout_penalty = -time_ratio * (1.0 - orientation_reward) * 0.05
    
    # Combine rewards with balanced weights
    total_reward = (
        orientation_reward * 2.5 +
        spin_reward * 1.0 +
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