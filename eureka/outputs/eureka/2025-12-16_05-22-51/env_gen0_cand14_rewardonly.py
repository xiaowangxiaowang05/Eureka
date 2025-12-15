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
        object_rot[:, 3:4] * goal_rot[:, 3:4] + torch.sum(object_rot[:, :3] * goal_rot[:, :3], dim=1, keepdim=True),
        object_rot[:, 3:4] * goal_rot[:, :3] - goal_rot[:, 3:4] * object_rot[:, :3] + torch.cross(object_rot[:, :3], goal_rot[:, :3], dim=1)
    ], dim=1)
    
    # Quaternion distance: 1 - |dot(q1, q2)|, but we use the angle via norm of xyz part
    # The scalar part of rel_quat is cos(theta/2), so sin(theta/2) = norm(xyz)
    angle_error = 2.0 * torch.asin(torch.clamp(torch.norm(rel_quat[:, 1:], dim=1), min=0.0, max=1.0))
    
    # Orientation reward: higher when angle error is smaller
    orientation_temp = 2.0
    orientation_reward = torch.exp(-orientation_temp * angle_error)
    
    # Angular velocity regularization: encourage some spinning but not too much
    # Target angular velocity magnitude (tuned for typical spinning tasks)
    target_angvel_mag = 2.0
    angvel_mag = torch.norm(object_angvel, dim=1)
    angvel_error = torch.abs(angvel_mag - target_angvel_mag)
    angvel_temp = 1.0
    angvel_reward = torch.exp(-angvel_temp * angvel_error)
    
    # Action regularization: penalize large actions for smoothness
    action_penalty = -torch.sum(actions ** 2, dim=1)
    action_weight = 0.01
    
    # Time-based survival bonus to encourage staying alive longer
    survival_bonus = 0.05 * (progress_buf / max_episode_length)
    
    # Combine rewards
    total_reward = (
        orientation_reward +
        0.5 * angvel_reward +
        action_weight * action_penalty +
        survival_bonus
    )
    
    reward_components = {
        "orientation_reward": orientation_reward,
        "angvel_reward": angvel_reward,
        "action_penalty": action_penalty,
        "survival_bonus": survival_bonus
    }
    
    return total_reward, reward_components