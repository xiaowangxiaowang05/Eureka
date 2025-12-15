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
    rot_error_angle = 2.0 * torch.asin(torch.clamp(torch.norm(rel_quat[:, :3], dim=-1), max=1.0))
    
    # Dense orientation reward: higher when aligned
    rot_reward = 1.0 / (rot_error_angle + 0.1)  # Avoid division by zero; ~[0.9 to 10] but we scale below

    # Encourage moderate spinning (not too fast, not too slow) — task is spin TO orientation, not maintain spin
    # So once aligned, low angvel is fine; during alignment, some motion needed
    # We don't penalize angvel directly but rely on action/joint costs for smoothness
    # Instead, we ensure the agent isn't stuck: if no progress, discourage idling via small negative if static
    # But main signal is rot_reward

    # Light action regularization to prevent jitter, but not so strong it freezes
    action_penalty = torch.sum(actions ** 2, dim=-1)
    
    # Joint velocity penalty only to smooth motion, very light
    joint_vel_penalty = torch.sum(shadow_hand_dof_vel ** 2, dim=-1)

    # Scale components to similar magnitudes
    total_reward = (
        1.0 * rot_reward
        - 0.05 * action_penalty
        - 0.001 * joint_vel_penalty
    )

    reward_components = {
        "rot_reward": rot_reward,
        "action_penalty": -0.05 * action_penalty,
        "joint_vel_penalty": -0.001 * joint_vel_penalty,
    }

    return total_reward, reward_components

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