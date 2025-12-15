def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    progress_buf: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error using quaternion distance
    # quat_mul(q1, quat_conjugate(q2)) gives relative rotation from q2 to q1
    rot_error_quat = quat_mul(object_rot, quat_conjugate(goal_rot))
    # Extract angle of rotation (magnitude of error)
    rot_error_angle = 2.0 * torch.asin(torch.clamp(torch.norm(rot_error_quat[:, 1:], dim=-1), min=0.0, max=1.0))
    
    # Orientation reward: higher when closer to target orientation
    orientation_temp = 1.0
    orientation_reward = torch.exp(-orientation_temp * rot_error_angle)
    
    # Angular velocity regularization: encourage smooth spinning, not too fast or too slow
    # Target angular velocity magnitude - we don't have explicit target, so just regularize extreme values
    angvel_norm = torch.norm(object_angvel, dim=-1)
    angvel_temp = 0.1
    angvel_reward = torch.exp(-angvel_temp * angvel_norm)
    
    # Action regularization: penalize large actions for energy efficiency
    action_norm = torch.norm(actions, dim=-1)
    action_temp = 0.01
    action_penalty = -action_temp * action_norm
    
    # Combine rewards
    total_reward = orientation_reward + 0.1 * angvel_reward + action_penalty
    
    reward_components = {
        "orientation_reward": orientation_reward,
        "angvel_reward": angvel_reward,
        "action_penalty": action_penalty
    }
    
    return total_response, reward_components

# Helper functions required for torch.jit.script compatibility
@torch.jit.script
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[:, 0:1], -q[:, 1:]], dim=-1)

@torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return torch.stack([w, x, y, z], dim=-1)