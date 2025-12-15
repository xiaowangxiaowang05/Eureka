def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    shadow_hand_dof_vel: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error using quaternion distance
    # quat_mul(q1, quat_conjugate(q2)) gives relative rotation from q2 to q1
    rel_rot = quat_mul(object_rot, quat_conjugate(goal_rot))
    # Extract angle-axis representation; the scalar part (w) relates to cosine of half-angle
    # Distance is 1 - |w|, but better to use full norm-based distance
    rot_error = 1.0 - torch.abs(rel_rot[:, 3])  # w component
    # Alternative: use norm of vector part
    rot_vec_norm = torch.norm(rel_rot[:, :3], dim=-1)
    rot_dist = rot_vec_norm  # ranges from 0 to 1
    
    # Temperature for orientation reward shaping
    rot_temp = 1.0
    rot_reward = torch.exp(-rot_temp * rot_dist)

    # Encourage appropriate angular velocity for spinning
    # Assume we want non-zero spin aligned with some axis? But task just says "target orientation"
    # Since it's orientation only (not continuous spin), we actually want LOW angular velocity at target
    angvel_norm = torch.norm(object_angvel, dim=-1)
    angvel_temp = 2.0
    angvel_penalty = torch.exp(-angvel_temp * angvel_norm)
    
    # Action regularization (smoothness/energy)
    action_norm = torch.norm(actions, dim=-1)
    action_temp = 0.05
    action_penalty = torch.exp(-action_temp * action_norm)
    
    # Joint velocity regularization to prevent jitter
    joint_vel_norm = torch.norm(shadow_hand_dof_vel, dim=-1)
    joint_vel_temp = 0.01
    joint_vel_penalty = torch.exp(-joint_vel_temp * joint_vel_norm)

    # Combine rewards
    reward = (
        2.0 * rot_reward +
        1.0 * angvel_penalty +
        0.5 * action_penalty +
        0.5 * joint_vel_penalty
    )

    reward_components = {
        "rot_reward": rot_reward,
        "angvel_penalty": angvel_penalty,
        "action_penalty": action_penalty,
        "joint_vel_penalty": joint_vel_penalty
    }

    return reward, reward_components


# Helper functions needed for TorchScript compatibility
@torch.jit.script
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([-q[:, :3], q[:, 3:4]], dim=-1)

@torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z2 * x1 - x2 * z1
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return torch.stack([x, y, z, w], dim=-1)