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
    # quat_mul(q1, quat_conjugate(q2)) gives relative rotation from q2 to q1
    rel_quat = torch.cat([
        object_rot[:, 3:4] * goal_rot[:, 0:3] - object_rot[:, 0:3] * goal_rot[:, 3:4] +
        torch.cross(object_rot[:, 0:3], goal_rot[:, 0:3], dim=-1),
        (object_rot[:, 3:4] * goal_rot[:, 3:4] + 
         torch.sum(object_rot[:, 0:3] * goal_rot[:, 0:3], dim=-1, keepdim=True))
    ], dim=-1)
    
    # Normalize the relative quaternion
    rel_quat_norm = torch.norm(rel_quat, dim=-1, keepdim=True)
    rel_quat = rel_quat / (rel_quat_norm + 1e-8)
    
    # Angular distance: ||log(q)|| = 2*arcsin(|imag(q)|)
    angle_error = 2.0 * torch.asin(torch.clamp(torch.norm(rel_quat[:, 0:3], dim=-1), min=0.0, max=1.0))
    
    # Temperature for orientation reward shaping
    orientation_temp = 1.0
    orientation_reward = torch.exp(-orientation_temp * angle_error)
    
    # Spin maintenance: encourage maintaining angular velocity aligned with desired spin axis
    # Assuming desired spin is around Z-axis of the object (can be generalized if needed)
    spin_axis = torch.zeros_like(object_angvel)
    spin_axis[:, 2] = 1.0  # Z-axis
    
    # Project angular velocity onto spin axis
    spin_proj = torch.sum(object_angvel * spin_axis, dim=-1)
    
    # Temperature for spin reward
    spin_temp = 0.5
    spin_reward = torch.exp(-spin_temp * torch.abs(spin_proj - 5.0))  # Target spin rate ~5 rad/s
    
    # Action regularization to minimize energy usage
    action_penalty = -torch.sum(actions ** 2, dim=-1) * 0.01
    
    # Early termination penalty (not used here since no explicit termination condition given)
    # But we can add a small penalty for being far from goal over time
    time_factor = (progress_buf / max_episode_length).float()
    timeout_penalty = -time_factor * (1.0 - orientation_reward) * 0.1
    
    # Combine rewards
    total_reward = (
        orientation_reward * 2.0 +
        spin_reward * 1.0 +
        action_penalty +
        timeout_penalty
    )
    
    reward_components = {
        "orientation_reward": orientation_reward,
        "spin_reward": spin_reward,
        "action_penalty": action_penalty,
        "timeout_penalty": timeout_penalty
    }
    
    return total_reward, reward_components