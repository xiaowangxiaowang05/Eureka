def compute_reward(
    object_pos: torch.Tensor,
    goal_pos: torch.Tensor,
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    shadow_hand_dof_vel: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Geodesic distance between object and goal rotation (0 to pi)
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    rot_error = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, :3], p=2, dim=-1), max=1.0))
    
    # Dense orientation reward
    rot_reward = torch.exp(-2.0 * rot_error)
    
    # Object drop penalty (fall_dist is typically ~0.1 in env, goal z ~0.5+0.1-0.04=0.56, so safe z > 0.1)
    drop_height_threshold = 0.1
    is_dropped = object_pos[:, 2] < drop_height_threshold
    drop_penalty = torch.where(is_dropped, -5.0, torch.zeros_like(rot_error))
    
    # Action regularization (very light)
    action_penalty = torch.sum(actions ** 2, dim=-1)
    action_reward = -0.005 * action_penalty

    # Light joint velocity penalty
    joint_vel_norm = torch.sum(shadow_hand_dof_vel ** 2, dim=-1)
    joint_vel_reward = -1e-6 * joint_vel_norm

    # Bonus for keeping object centered (prevents drifting)
    pos_error = torch.norm(object_pos - goal_pos, dim=-1)
    center_bonus = torch.exp(-5.0 * pos_error)
    
    total_reward = rot_reward + drop_penalty + action_reward + joint_vel_reward + center_bonus

    reward_components = {
        "rot_reward": rot_reward,
        "drop_penalty": drop_penalty,
        "action_reward": action_reward,
        "joint_vel_reward": joint_vel_reward,
        "center_bonus": center_bonus
    }

    return total_reward, reward_components

@torch.jit.script
def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[-1] == 4
    assert b.shape[-1] == 4

    x1, y1, z1, w1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    x2, y2, z2, w2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([x, y, z, w], dim=-1)

@torch.jit.script
def quat_conjugate(a: torch.Tensor) -> torch.Tensor:
    conj = torch.cat([-a[..., 0:3], a[..., 3:4]], dim=-1)
    return conj