def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    shadow_hand_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error using quaternion distance
    # quat_mul(q1, quat_conjugate(q2)) gives relative rotation from q2 to q1
    rel_quat = quat_mul(object_rot, quat_conjugate(goal_rot))
    # The scalar part (w component) indicates alignment; closer to 1 means better alignment
    orientation_error = 1.0 - rel_quat[:, 0]  # w component is at index 0
    orientation_reward_temp = 2.0
    orientation_reward = torch.exp(-orientation_error * orientation_reward_temp)

    # Angular velocity regularization: encourage smooth spinning without excessive speed
    angvel_norm = torch.norm(object_angvel, dim=-1)
    angvel_penalty_temp = 0.1
    angvel_penalty = torch.exp(-angvel_penalty_temp * angvel_norm)

    # Action regularization: penalize large control efforts
    action_norm = torch.norm(actions, dim=-1)
    action_penalty_temp = 0.01
    action_penalty = torch.exp(-action_penalty_temp * action_norm)

    # Joint velocity regularization: discourage jerky movements
    joint_vel_norm = torch.norm(shadow_hand_dof_vel, dim=-1)
    joint_vel_penalty_temp = 0.05
    joint_vel_penalty = torch.exp(-joint_vel_penalty_temp * joint_vel_norm)

    # Combine rewards with weights
    total_reward = (
        2.0 * orientation_reward +
        0.5 * angvel_penalty +
        0.2 * action_penalty +
        0.1 * joint_vel_penalty
    )

    reward_components = {
        "orientation_reward": orientation_reward,
        "angvel_penalty": angvel_penalty,
        "action_penalty": action_penalty,
        "joint_vel_penalty": joint_vel_penalty
    }

    return total_reward, reward_components


# Helper functions required for TorchScript compatibility
@torch.jit.script
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([-q[:, :3], q[:, 3:4]], dim=-1)

@torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return torch.stack([x, y, z, w], dim=-1)