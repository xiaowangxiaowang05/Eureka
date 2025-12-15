def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    shadow_hand_dof_vel: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # --- Distance from hand (fingertips) to object center is CRITICAL for grasp stability ---
    # However, fingertip positions aren't in inputs. Instead, we approximate "object not dropped"
    # by requiring object z > threshold (since drop = fall down due to gravity simulation)
    # But better: use angular velocity norm as proxy – very low angvel + wrong orientation = dropped.
    # We already have a frozen penalty, but it's too weak. Strengthen it and add exponential decay.

    # --- Orientation alignment reward ---
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, :3], dim=-1), max=1.0))
    rot_reward = torch.exp(-1.5 * rot_dist)

    # --- Spin reward (only if aligned) ---
    axis_norm = torch.norm(quat_diff[:, :3], dim=-1, keepdim=True)
    spin_axis = torch.where(axis_norm > 1e-6, quat_diff[:, :3] / axis_norm, torch.zeros_like(quat_diff[:, :3]))
    angvel_on_axis = torch.sum(object_angvel * spin_axis, dim=-1)
    angvel_mag = torch.abs(angvel_on_axis)
    target_spin = 4.0
    spin_error = torch.abs(angvel_mag - target_spin)
    spin_reward_raw = torch.exp(-0.4 * spin_error)
    spin_reward = torch.where(rot_dist < 0.8, spin_reward_raw, torch.zeros_like(spin_reward_raw))

    # --- Action regularization (light) ---
    action_penalty = torch.sum(actions ** 2, dim=-1)
    action_reward = -1e-4 * action_penalty

    joint_vel_penalty = torch.sum(shadow_hand_dof_vel ** 2, dim=-1)
    joint_vel_reward = -5e-8 * joint_vel_penalty

    # --- STRONG frozen/dropped object penalty ---
    angvel_norm = torch.norm(object_angvel, dim=-1)
    # If object is nearly stopped AND far from goal orientation, it's likely dropped
    frozen_object_penalty = torch.where(
        (angvel_norm < 0.6) & (rot_dist > 0.7),
        -3.0 * torch.ones_like(angvel_norm),
        torch.zeros_like(angvel_norm)
    )

    # --- Critical: Add dense "keep object" reward via angular velocity maintenance when misaligned ---
    # Encourage ANY motion when orientation is bad to avoid static failure
    movement_bonus = torch.where(
        rot_dist > 1.0,
        torch.clamp(angvel_norm * 0.1, max=0.5),
        torch.zeros_like(angvel_norm)
    )

    total_reward = rot_reward + spin_reward + action_reward + joint_vel_reward + frozen_object_penalty + movement_bonus

    reward_components = {
        "rot_reward": rot_reward,
        "spin_reward": spin_reward,
        "action_reward": action_reward,
        "joint_vel_reward": joint_vel_reward,
        "frozen_object_penalty": frozen_object_penalty,
        "movement_bonus": movement_bonus
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