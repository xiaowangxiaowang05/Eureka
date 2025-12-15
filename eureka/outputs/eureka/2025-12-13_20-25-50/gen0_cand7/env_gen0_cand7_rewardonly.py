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
    rel_quat = torch.mul(object_rot, quat_conjugate(goal_rot))
    # The w component of the relative quaternion relates to the angle difference
    # Orientation error: 1 - |w| (since |w| = cos(theta/2), so smaller angle => larger |w|)
    orient_error = 1.0 - torch.abs(rel_quat[:, 3])  # rel_quat[:, 3] is the w component
    
    # Exponential reward for orientation alignment
    orient_temp = 2.0
    orient_reward = torch.exp(-orient_temp * orient_error)

    # Encourage appropriate angular velocity for spinning
    # We don't know the exact axis, so we encourage sufficient magnitude
    angvel_norm = torch.norm(object_angvel, dim=-1)
    target_angvel = 2.0  # target angular velocity magnitude
    angvel_error = torch.abs(angvel_norm - target_angvel)
    angvel_temp = 1.0
    angvel_reward = torch.exp(-angvel_temp * angvel_error)

    # Action regularization to minimize control effort
    action_penalty = torch.sum(actions ** 2, dim=-1)
    action_temp = 0.01
    action_reward = torch.exp(-action_temp * action_penalty)

    # Time-based bonus to encourage solving task quickly
    time_bonus = progress_buf / max_episode_length

    # Combine rewards with weights
    total_reward = (
        2.0 * orient_reward +
        1.0 * angvel_reward +
        0.5 * action_reward +
        0.1 * time_bonus
    )

    reward_components = {
        "orient_reward": orient_reward,
        "angvel_reward": angvel_reward,
        "action_reward": action_reward,
        "time_bonus": time_bonus
    }

    return total_reward, reward_components

# Helper functions needed for torch.jit compatibility
@torch.jit.script
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([-q[:, :3], q[:, 3:4]], dim=-1)

@torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([x, y, z, w], dim=-1)