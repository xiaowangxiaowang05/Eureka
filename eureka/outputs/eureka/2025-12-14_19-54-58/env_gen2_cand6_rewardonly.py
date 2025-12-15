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
    
    # Spin reward: encourage sustained rotation around object's local Z-axis
    spin_axis = torch.zeros_like(object_angvel)
    spin_axis[:, 2] = 1.0
    spin_proj = torch.sum(object_angvel * spin_axis, dim=-1)
    spin_temp = 0.5
    spin_reward = torch.exp(-spin_temp * torch.abs(spin_proj - 5.0))
    
    # Action regularization: penalize large actions
    action_penalty = -torch.sum(actions ** 2, dim=-1) * 0.01
    
    # Action smoothness: penalize rapid changes in actions
    action_smoothness_penalty = -torch.sum((actions - prev_actions) ** 2, dim=-1) * 0.1
    
    # Joint acceleration penalty: approximate via change in velocity
    joint_accel_penalty = -torch.sum(shadow_hand_dof_vel ** 2, dim=-1) * 0.001
    
    # Default pose penalty: discourage deviation from natural hand posture
    default_pose_penalty = -torch.sum((shadow_hand_dof_pos - shadow_hand_default_dof_pos) ** 2, dim=-1) * 0.01
    
    # Timeout penalty to discourage stalling
    time_factor = (progress_buf / max_episode_length).float()
    timeout_penalty = -time_factor * (1.0 - orientation_reward) * 0.1
    
    total_reward = (
        orientation_reward * 2.0 +
        spin_reward * 1.0 +
        action_penalty +
        action_smoothness_penalty +
        joint_accel_penalty +
        default_pose_penalty +
        timeout_penalty
    )
    
    reward_components = {
        "orientation_reward": orientation_reward,
        "spin_reward": spin_reward,
        "action_penalty": action_penalty,
        "action_smoothness_penalty": action_smoothness_penalty,
        "joint_accel_penalty": joint_accel_penalty,
        "default_pose_penalty": default_pose_penalty,
        "timeout_penalty": timeout_penalty
    }
    
    return total_reward, reward_components