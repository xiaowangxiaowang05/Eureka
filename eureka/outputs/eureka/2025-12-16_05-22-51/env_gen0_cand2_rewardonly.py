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
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    # Extract angle-axis representation; angle = 2 * acos(w)
    w = quat_diff[:, 3]  # scalar part
    # Clamp to avoid numerical issues
    w = torch.clamp(w, -1.0, 1.0)
    angle_error = 2.0 * torch.acos(torch.abs(w))  # in [0, pi]

    # Orientation reward: higher when closer to target orientation
    orientation_temp = 1.0
    orientation_reward = torch.exp(-orientation_temp * angle_error)

    # Angular velocity regularization: encourage smooth spinning, not too fast
    angvel_norm = torch.norm(object_angvel, dim=-1)
    angvel_temp = 0.5
    angvel_reward = torch.exp(-angvel_temp * angvel_norm)

    # Action regularization: penalize large actions for energy efficiency
    action_penalty = torch.sum(actions ** 2, dim=-1)
    action_temp = 0.01
    action_reward = -action_temp * action_penalty

    # Time-based bonus: slightly encourage solving faster (optional but helpful)
    time_bonus = 0.01 * (1.0 - (progress_buf / max_episode_length))

    # Total reward
    total_reward = orientation_reward + 0.1 * angvel_reward + action_reward + time_bonus

    reward_components = {
        "orientation_reward": orientation_reward,
        "angvel_reward": angvel_reward,
        "action_reward": action_reward,
        "time_bonus": time_bonus
    }

    return total_reward, reward_components


# Helper functions required for TorchScript compatibility
@torch.jit.script
def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape == b.shape
    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return torch.stack([x, y, z, w], dim=-1)

@torch.jit.script
def quat_conjugate(a: torch.Tensor) -> torch.Tensor:
    return torch.cat([-a[:, :3], a[:, 3:4]], dim=-1)