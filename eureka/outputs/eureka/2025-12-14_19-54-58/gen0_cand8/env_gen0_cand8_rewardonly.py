def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    shadow_hand_dof_vel: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error using quaternion distance
    # quat_mul(q1, quat_conjugate(q2)) gives relative rotation from q2 to q1
    rel_quat = quat_mul(object_rot, quat_conjugate(goal_rot))
    # Extract angle-axis representation; angle = 2 * acos(|w|)
    w = rel_quat[:, 3]  # scalar part of quaternion
    # Clamp for numerical stability
    w = torch.clamp(w, -1.0, 1.0)
    rot_error_angle = 2.0 * torch.acos(torch.abs(w))
    
    # Temperature parameter for orientation reward shaping
    rot_temp = 1.0
    rot_reward = torch.exp(-rot_temp * rot_error_angle)

    # Encourage appropriate angular velocity aligned with task needs
    # We assume spinning around any axis is fine as long as orientation matches,
    # but we penalize excessive angular speed that might destabilize
    angvel_norm = torch.norm(object_angvel, dim=-1)
    angvel_temp = 0.5
    angvel_reward = torch.exp(-angvel_temp * angvel_norm)

    # Action regularization: penalize large actions for energy efficiency
    action_penalty = torch.sum(actions ** 2, dim=-1)
    action_temp = 0.01
    action_reward = torch.exp(-action_temp * action_penalty)

    # Joint velocity regularization to prevent jittery motions
    joint_vel_penalty = torch.sum(shadow_hand_dof_vel ** 2, dim=-1)
    joint_vel_temp = 0.001
    joint_vel_reward = torch.exp(-joint_vel_temp * joint_vel_penalty)

    # Combine rewards with weights
    total_reward = (
        2.0 * rot_reward +
        0.5 * angvel_reward +
        0.3 * action_reward +
        0.2 * joint_vel_reward
    )

    reward_components = {
        "rot_reward": rot_reward,
        "angvel_reward": angvel_reward,
        "action_reward": action_reward,
        "joint_vel_reward": joint_vel_reward,
    }

    return total_reward, reward_components

# Helper functions required for TorchScript
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