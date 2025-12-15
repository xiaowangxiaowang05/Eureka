def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    shadow_hand_dof_vel: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute rotational error using quaternion difference
    rel_quat = quat_mul(object_rot, quat_conjugate(goal_rot))
    rot_error = 2.0 * torch.asin(torch.clamp(torch.norm(rel_quat[:, :3], dim=-1), max=1.0))
    
    # Dense orientation reward: higher when close to target
    rot_temp = 2.0
    rot_reward = torch.exp(-rot_temp * rot_error)

    # Encourage controlled spinning: moderate angular velocity is okay, but penalize excessive spin
    angvel_norm = torch.norm(object_angvel, dim=-1)
    angvel_limit = 5.0  # reasonable spinning speed
    angvel_penalty = torch.abs(angvel_norm - angvel_limit)
    angvel_temp = 0.5
    angvel_reward = torch.exp(-angvel_temp * angvel_penalty)

    # Light action regularization to avoid jitter, but not so strong it prevents motion
    action_penalty = torch.sum(actions ** 2, dim=-1)
    action_scale = 0.01
    action_reward = -action_scale * action_penalty

    # Very light joint velocity penalty to discourage high-frequency noise
    joint_vel_penalty = torch.sum(shadow_hand_dof_vel ** 2, dim=-1)
    joint_vel_scale = 0.0001
    joint_vel_reward = -joint_vel_scale * joint_vel_penalty

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

@torch.jit.script
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    conj = torch.cat([-q[:, :3], q[:, 3:4]], dim=-1)
    return conj

@torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return torch.stack([x, y, z, w], dim=-1)