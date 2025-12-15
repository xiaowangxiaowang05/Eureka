def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    shadow_hand_dof_vel: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute relative rotation quaternion
    rel_quat = quat_mul(object_rot, quat_conjugate(goal_rot))
    # Compute shortest arc angle (in [0, pi])
    rot_error_angle = 2.0 * torch.acos(torch.clamp(torch.abs(rel_quat[:, 3]), min=0.0, max=1.0))
    
    # Dense reward for orientation alignment (higher when closer)
    rot_reward_scale = 2.0
    rot_reward = rot_reward_scale * (1.0 - rot_error_angle / torch.pi)

    # Small bonus for smooth angular motion (not zero but not excessive)
    angvel_norm = torch.norm(object_angvel, dim=-1)
    angvel_target = 1.0  # encourage some spin, not too fast or too slow
    angvel_error = torch.abs(angvel_norm - angvel_target)
    angvel_temp = 0.5
    angvel_reward = torch.exp(-angvel_temp * angvel_error)

    # Light action regularization to avoid jitter, but not so strong it prevents motion
    action_penalty = torch.sum(actions ** 2, dim=-1)
    action_reward = -0.01 * action_penalty

    # Very light joint velocity penalty to discourage extreme oscillations
    joint_vel_penalty = torch.sum(shadow_hand_dof_vel ** 2, dim=-1)
    joint_vel_reward = -0.0001 * joint_vel_penalty

    total_reward = (
        rot_reward +
        0.3 * angvel_reward +
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

# Re-declare helper functions for TorchScript compatibility within the same scope
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