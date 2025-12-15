def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    progress_buf: torch.Tensor,
    max_episode_length: float
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error using quaternion distance
    # quat_mul(object_rot, quat_conjugate(goal_rot)) gives relative rotation
    # The scalar part (w) of this relative rotation indicates alignment
    rel_quat = quat_mul(object_rot, quat_conjugate(goal_rot))
    orientation_error = 1.0 - torch.abs(rel_quat[:, 0])  # 0 when aligned, up to 1 when opposite
    
    # Exponential reward for orientation alignment
    orientation_temp = 2.0
    orientation_reward = torch.exp(-orientation_temp * orientation_error)
    
    # Encourage maintaining some angular velocity to keep spinning
    angvel_norm = torch.norm(object_angvel, dim=-1)
    target_angvel = 2.0  # target magnitude of angular velocity
    angvel_error = torch.abs(angvel_norm - target_angvel)
    angvel_temp = 1.0
    angvel_reward = torch.exp(-angvel_temp * angvel_error)
    
    # Action regularization to discourage excessive motor effort
    action_penalty = torch.sum(actions ** 2, dim=-1)
    action_temp = 0.01
    action_reward = -action_temp * action_penalty
    
    # Time-based bonus to encourage faster solving (only near the end)
    time_bonus = (progress_buf / max_episode_length) * 0.1
    
    # Combine rewards
    total_reward = orientation_reward + 0.5 * angvel_reward + action_reward + time_bonus
    
    reward_components = {
        "orientation_reward": orientation_reward,
        "angvel_reward": angvel_reward,
        "action_reward": action_reward,
        "time_bonus": time_bonus
    }
    
    return total_reward, reward_components

# Helper functions required for torch.jit.script compatibility
@torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    assert q1.shape[-1] == 4 and q2.shape[-1] == 4
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([x, y, z, w], dim=-1)

@torch.jit.script
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([-q[..., :3], q[..., 3:4]], dim=-1)