def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    dof_vel: torch.Tensor,
    progress_buf: torch.Tensor,
    max_episode_length: float
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error using quaternion distance
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    angle_error = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:], dim=-1), min=0.0, max=1.0))
    
    # Temperature parameter for orientation reward shaping
    orientation_temp = 2.0
    orientation_reward = torch.exp(-orientation_temp * angle_error)
    
    # Encourage appropriate angular velocity magnitude (not too slow, not too fast)
    angvel_norm = torch.norm(object_angvel, dim=-1)
    target_angvel_mag = 3.0  # target spinning speed
    angvel_error = torch.abs(angvel_norm - target_angvel_mag)
    angvel_temp = 0.5
    angvel_reward = torch.exp(-angvel_temp * angvel_error)
    
    # Action regularization to encourage energy efficiency
    action_penalty = -torch.sum(actions ** 2, dim=-1) * 0.0001
    
    # Joint velocity regularization to prevent jitter/motor strain
    joint_vel_penalty = -torch.sum(dof_vel ** 2, dim=-1) * 0.00001
    
    # Success bonus for being very close to target orientation
    success_threshold = 0.1  # radians
    success_bonus = torch.where(angle_error < success_threshold, 
                               5.0 * (1.0 - progress_buf / max_episode_length),
                               torch.zeros_like(angle_error))
    
    # Combine rewards
    total_reward = (
        orientation_reward +
        angvel_reward +
        action_penalty +
        joint_vel_penalty +
        success_bonus
    )
    
    reward_components = {
        "orientation_reward": orientation_reward,
        "angvel_reward": angvel_reward,
        "action_penalty": action_penalty,
        "joint_vel_penalty": joint_vel_penalty,
        "success_bonus": success_bonus
    }
    
    return total_reward, reward_components

# Helper functions required for TorchScript compatibility
@torch.jit.script
def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[-1] == 4
    assert b.shape[-1] == 4
    
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return torch.stack([w, x, y, z], dim=-1)

@torch.jit.script
def quat_conjugate(a: torch.Tensor) -> torch.Tensor:
    return torch.cat([-a[..., :1], a[..., 1:]], dim=-1)