def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    shadow_hand_dof_vel: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error using quaternion distance
    # quat_mul(q1, quat_conjugate(q2)) gives relative rotation from q2 to q1
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    # Extract angle-axis representation; angle = 2 * acos(w)
    rot_error = 1.0 - torch.abs(quat_diff[:, 3])  # w component; |w| close to 1 means aligned
    rot_error = torch.clamp(rot_error, 0.0, 1.0)
    
    # Temperature for orientation reward shaping
    rot_temp: float = 2.0
    rot_reward = torch.exp(-rot_temp * rot_error)

    # Encourage appropriate angular velocity (not too slow, not too fast)
    angvel_norm = torch.norm(object_angvel, dim=-1)
    target_angvel: float = 2.0  # target spinning speed in rad/s
    angvel_error = torch.abs(angvel_norm - target_angvel)
    angvel_temp: float = 0.5
    angvel_reward = torch.exp(-angvel_temp * angvel_error)

    # Action regularization to reduce jitter and energy use
    action_penalty = torch.sum(actions ** 2, dim=-1)
    action_temp: float = 0.01
    action_reward = -action_temp * action_penalty

    # Joint velocity regularization to prevent excessive motion
    joint_vel_penalty = torch.sum(shadow_hand_dof_vel ** 2, dim=-1)
    joint_vel_temp: float = 0.001
    joint_vel_reward = -joint_vel_temp * joint_vel_penalty

    # Combine rewards
    total_reward = (
        rot_reward +
        0.5 * angvel_reward +
        action_reward +
        joint_vel_reward
    )

    reward_components = {
        "rot_reward": rot_reward,
        "angvel_reward": angvel_reward,
        "action_reward": action_reward,
        "joint_vel_reward": joint_vel_reward,
    }

    return total_reward, reward_components


# Helper functions required for TorchScript compatibility
@torch.jit.script
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([-q[:, :3], q[:, 3:4]], dim=-1)

@torch.jit.script
def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (y1 - z1) * (x2 - w2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return torch.stack([x, y, z, w], dim=-1)