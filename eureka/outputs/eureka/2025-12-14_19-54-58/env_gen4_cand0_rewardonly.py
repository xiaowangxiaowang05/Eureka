@torch.jit.script
def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    progress_buf: torch.Tensor,
    max_episode_length: float,
    fingertip_pos: torch.Tensor,
    object_pos: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Orientation error
    rel_quat = quat_mul(object_rot, quat_conjugate(goal_rot))
    angle_error = 2.0 * torch.asin(torch.clamp(torch.norm(rel_quat[:, 0:3], dim=-1), min=0.0, max=1.0))
    orientation_temp = 1.5
    orientation_reward = torch.exp(-orientation_temp * angle_error)
    
    # Spin reward: target ~4 rad/s around object z-axis (less aggressive than 5)
    spin_axis = torch.zeros_like(object_angvel)
    spin_axis[:, 2] = 1.0
    spin_proj = torch.sum(object_angvel * spin_axis, dim=-1)
    spin_temp = 0.4
    spin_reward = torch.exp(-spin_temp * torch.abs(spin_proj - 4.0))
    
    # Fingertip proximity reward: encourage fingers near object
    fingertip_object_dist = torch.norm(fingertip_pos - object_pos.unsqueeze(1), dim=-1)  # [num_envs, 5]
    mean_fingertip_dist = torch.mean(fingertip_object_dist, dim=-1)
    proximity_temp = 2.0
    proximity_reward = torch.exp(-proximity_temp * mean_fingertip_dist)
    
    # Hand-object distance penalty (coarse grasp encouragement)
    hand_object_dist = torch.norm(object_pos - torch.mean(fingertip_pos, dim=1), dim=-1)
    dist_penalty = -hand_object_dist * 0.5
    
    # Reduced action penalty to avoid freezing
    action_norm = torch.sum(actions ** 2, dim=-1)
    action_penalty = -action_norm * 0.02
    
    # Moderate angular velocity penalty to prevent flinging
    angvel_penalty = -torch.norm(object_angvel, dim=-1) * 0.05
    
    # Small timeout penalty
    time_ratio = (progress_buf / max_episode_length).float()
    timeout_penalty = -time_ratio * angle_error * 0.05
    
    # Combine rewards with balanced weights
    total_reward = (
        orientation_reward * 2.5 +
        spin_reward * 1.0 +
        proximity_reward * 1.5 +
        dist_penalty +
        action_penalty +
        angvel_penalty +
        timeout_penalty
    )
    
    reward_components = {
        "orientation_reward": orientation_reward,
        "spin_reward": spin_reward,
        "proximity_reward": proximity_reward,
        "dist_penalty": dist_penalty,
        "action_penalty": action_penalty,
        "angvel_penalty": angvel_penalty,
        "timeout_penalty": timeout_penalty
    }
    
    return total_reward, reward_components