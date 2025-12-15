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
    rel_quat = rel_quat / torch.clamp(rel_quat_norm, min=1e-6)
    
    # Angular difference (angle of rotation needed to align orientations)
    angle_error = 2.0 * torch.asin(torch.clamp(torch.abs(rel_quat[:, 3]), min=0.0, max=1.0))
    
    # Orientation reward: encourage small angular error
    orientation_temp = 1.0
    orientation_reward = torch.exp(-orientation_temp * angle_error)
    
    # Encourage spinning behavior by rewarding angular velocity magnitude
    # This helps the agent learn to spin the object rather than just statically hold it
    angvel_magnitude = torch.norm(object_angvel, dim=-1)
    spin_temp = 0.1
    spin_reward = torch.exp(-spin_temp * torch.abs(angvel_magnitude - 5.0))  # Target ~5 rad/s
    
    # Action regularization to minimize energy usage
    action_penalty = torch.sum(actions ** 2, dim=-1)
    action_temp = 0.01
    action_reward = -action_temp * action_penalty
    
    # Success bonus for very close alignment (dense shaping near success)
    success_bonus = (angle_error < 0.1).float() * 2.0
    
    # Combine rewards
    total_reward = orientation_reward + spin_reward + action_reward + success_bonus
    
    reward_components = {
        "orientation_reward": orientation_reward,
        "spin_reward": spin_reward,
        "action_reward": action_reward,
        "success_bonus": success_bonus
    }
    
    return total_reward, reward_components