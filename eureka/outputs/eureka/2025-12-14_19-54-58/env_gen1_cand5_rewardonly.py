@torch.jit.script
def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    progress_buf: torch.Tensor,
    max_episode_length: float,
    shadow_hand_dof_pos: torch.Tensor,
    shadow_hand_dof_vel: torch.Tensor,
    prev_actions: torch.Tensor,
    shadow_hand_default_dof_pos: torch.Tensor
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
    
    # Smoothness penalty (change in actions)
    action_delta = actions - prev_actions
    smoothness_penalty = -torch.sum(action_delta ** 2, dim=-1) * 0.05
    
    # Joint acceleration penalty (approximated via velocity difference)
    # Since we don't have previous velocity in inputs, we approximate with current velocity magnitude
    acceleration_penalty = -torch.sum(shadow_hand_dof_vel ** 2, dim=-1) * 0.005
    
    # Default pose deviation penalty
    default_pose_error = torch.norm(shadow_hand_dof_pos - shadow_hand_default_dof_pos, dim=-1)
    default_pose_penalty = -default_pose_error * 0.02
    
    # Timeout penalty
    time_factor = (progress_buf / max_episode_length).float()
    timeout_penalty = -time_factor * (1.0 - orientation_reward) * 0.1
    
    total_reward = (
        orientation_reward * 2.0 +
        spin_reward * 1.0 +
        action_penalty +
        smoothness_penalty +
        acceleration_penalty +
        default_pose_penalty +
        timeout_penalty
    )
    
    reward_components = {
        "orientation_reward": orientation_reward,
        "spin_reward": spin_reward,
        "action_penalty": action_penalty,
        "smoothness_penalty": smoothness_penalty,
        "acceleration_penalty": acceleration_penalty,
        "default_pose_penalty": default_pose_penalty,
        "timeout_penalty": timeout_penalty
    }
    
    return total_reward, reward_components