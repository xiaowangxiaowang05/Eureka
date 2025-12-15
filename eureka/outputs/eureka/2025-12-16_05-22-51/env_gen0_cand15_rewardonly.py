def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    shadow_hand_dof_vel: torch.Tensor,
    progress_buf: torch.Tensor,
    max_episode_length: float
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error using quaternion distance
    # quat_mul(q1, quat_conjugate(q2)) gives relative rotation from q2 to q1
    rel_quat = quat_mul(object_rot, quat_conjugate(goal_rot))
    # The scalar part (w) of the relative quaternion indicates alignment
    # Alignment is perfect when rel_quat[..., 0] = ±1
    rot_error = 1.0 - torch.abs(rel_quat[..., 0])
    
    # Temperature for orientation reward shaping
    rot_temp = 2.0
    rot_reward = torch.exp(-rot_temp * rot_error)
    
    # Encourage maintaining some angular velocity to keep spinning (prevent static hold)
    # But don't reward excessive spin
    angvel_norm = torch.norm(object_angvel, dim=-1)
    angvel_target = 2.0  # target angular speed in rad/s
    angvel_error = torch.abs(angvel_norm - angvel_target)
    angvel_temp = 0.5
    angvel_reward = torch.exp(-angvel_temp * angvel_error)
    
    # Action regularization: penalize large actions
    action_penalty = torch.sum(actions ** 2, dim=-1)
    action_temp = 0.01
    action_reward = -action_temp * action_penalty
    
    # Joint velocity regularization: penalize very fast joint movements
    joint_vel_penalty = torch.sum(shadow_hand_dof_vel ** 2, dim=-1)
    joint_vel_temp = 0.0001
    joint_vel_reward = -joint_vel_temp * joint_vel_penalty
    
    # Time-based survival bonus to encourage longer episodes (helps exploration)
    # Only apply during early training; diminishes as episode progresses
    time_ratio = progress_buf / max_episode_length
    survival_bonus = 0.1 * (1.0 - time_ratio)
    
    # Combine rewards
    total_reward = (
        rot_reward +
        0.5 * angvel_reward +
        action_reward +
        joint_vel_reward +
        survival_bonus
    )
    
    reward_components = {
        "rot_reward": rot_reward,
        "angvel_reward": angvel_reward,
        "action_reward": action_reward,
        "joint_vel_reward": joint_vel_reward,
        "survival_bonus": survival_bonus
    }
    
    return total_reward, reward_components

# Helper functions required for TorchScript
@torch.jit.script
def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[-1] == 4
    assert b.shape[-1] == 4

    x1, y1, z1, w1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    x2, y2, z2, w2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

    return torch.stack([x, y, z, w], dim=-1)

@torch.jit.script
def quat_conjugate(a: torch.Tensor) -> torch.Tensor:
    return torch.cat([-a[..., :3], a[..., 3:]], dim=-1)