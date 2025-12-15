def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    shadow_hand_dof_vel: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error using quaternion distance
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    rot_dist = 1.0 - quat_diff[:, 3]  # [0, 2], lower is better

    # Dense orientation reward (higher weight)
    rot_reward_scale = 3.0
    rot_reward = rot_reward_scale * (1.0 - rot_dist)

    # Encourage angular velocity aligned with target spin axis
    # Assume target is static: we want zero angvel when aligned, but some motion during transition
    # However, task is "spin to target orientation", so once aligned, minimal motion is fine
    # But initial phases need motion → use |angvel| only as mild penalty to prevent instability
    angvel_penalty = torch.norm(object_angvel, dim=-1)
    angvel_penalty_scale = 0.1
    angvel_reward = -angvel_penalty_scale * angvel_penalty

    # Small action regularization (linear, not exponential)
    action_penalty = torch.sum(torch.abs(actions), dim=-1)
    action_penalty_scale = 0.05
    action_reward = -action_penalty_scale * action_penalty

    # Small joint velocity regularization
    joint_vel_penalty = torch.sum(torch.abs(shadow_hand_dof_vel), dim=-1)
    joint_vel_penalty_scale = 0.01
    joint_vel_reward = -joint_vel_penalty_scale * joint_vel_penalty

    # Total reward
    total_reward = rot_reward + angvel_reward + action_reward + joint_vel_reward

    reward_components = {
        "rot_reward": rot_reward,
        "angvel_reward": angvel_reward,
        "action_reward": action_reward,
        "joint_vel_reward": joint_vel_reward
    }

    return total_reward, reward_components

# Helper functions required for TorchScript compatibility
@torch.jit.script
def quat_conjugate(q):
    return torch.cat([-q[:, :3], q[:, 3:4]], dim=-1)

@torch.jit.script
def quat_mul(q1, q2):
    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([x, y, z, w], dim=-1)