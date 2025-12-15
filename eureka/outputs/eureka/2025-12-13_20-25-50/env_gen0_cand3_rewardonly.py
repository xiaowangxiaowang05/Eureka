def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    shadow_hand_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error using quaternion distance
    # quat_mul(q1, quat_conjugate(q2)) gives relative rotation from q2 to q1
    rel_rot = quat_mul(object_rot, quat_conjugate(goal_rot))
    # Extract angle-axis representation; angle is 2*acos(w)
    w = rel_rot[:, 3]  # scalar part of quaternion
    # Clamp w to [-1, 1] to avoid NaNs due to numerical errors
    w = torch.clamp(w, -1.0, 1.0)
    orientation_error = 2.0 * torch.acos(torch.abs(w))  # angular error in [0, pi]

    # Temperature for orientation reward shaping
    orientation_temp = 1.0
    orientation_reward = torch.exp(-orientation_temp * orientation_error)

    # Encourage appropriate angular velocity magnitude (not too slow, not too fast)
    # Target a moderate spin speed to avoid instability
    angvel_norm = torch.norm(object_angvel, dim=-1)
    target_angvel = 2.0  # rad/s
    angvel_error = torch.abs(angvel_norm - target_angvel)
    angvel_temp = 0.5
    angvel_reward = torch.exp(-angvel_temp * angvel_error)

    # Action regularization: penalize large actions
    action_penalty = torch.sum(actions**2, dim=-1)
    action_reg_weight = 0.01
    action_reward = -action_reg_weight * action_penalty

    # Joint velocity regularization to prevent jittery movements
    joint_vel_penalty = torch.sum(shadow_hand_dof_vel**2, dim=-1)
    joint_vel_weight = 0.001
    joint_vel_reward = -joint_vel_weight * joint_vel_penalty

    # Combine rewards
    total_reward = (
        orientation_reward +
        0.5 * angvel_reward +
        action_reward +
        joint_vel_reward
    )

    reward_components = {
        "orientation_reward": orientation_reward,
        "angvel_reward": angvel_reward,
        "action_reward": action_reward,
        "joint_vel_reward": joint_vel_reward
    }

    return total_reward, reward_components


# Helper functions required for torch.jit compatibility
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
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([-q[..., :3], q[..., 3:]], dim=-1)