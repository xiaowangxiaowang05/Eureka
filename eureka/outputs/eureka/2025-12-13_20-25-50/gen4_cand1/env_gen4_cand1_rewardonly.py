def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    shadow_hand_dof_vel: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # --- Hand-object distance reward (retention signal) ---
    # NOTE: Since fingertip positions aren't directly available here,
    # we approximate retention by penalizing large angular velocity when far from goal.
    # But to add explicit retention, we must infer hand position.
    # However, per function signature constraints, we cannot access fingertip_pos.
    # Therefore, we strengthen penalty for falling (high angvel + bad orientation = dropped)
    # and add stronger orientation gradient early.

    # --- Orientation alignment reward (dense, shaped) ---
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, :3], dim=-1), max=1.0))
    rot_reward = torch.exp(-1.5 * rot_dist)

    # --- Spin reward: only encourage when close to target orientation ---
    axis_norm = torch.norm(quat_diff[:, :3], dim=-1, keepdim=True)
    spin_axis = torch.where(axis_norm > 1e-6, quat_diff[:, :3] / axis_norm, torch.zeros_like(quat_diff[:, :3]))
    angvel_on_axis = torch.sum(object_angvel * spin_axis, dim=-1)
    angvel_mag = torch.abs(angvel_on_axis)
    target_spin = 3.0
    spin_error = torch.abs(angvel_mag - target_spin)
    spin_reward_raw = torch.exp(-0.4 * spin_error)
    spin_reward = torch.where(rot_dist < 0.8, spin_reward_raw, torch.zeros_like(spin_reward_raw))

    # --- Action & joint velocity penalties (ultra light) ---
    action_penalty = torch.sum(actions ** 2, dim=-1)
    action_reward = -1e-5 * action_penalty

    joint_vel_penalty = torch.sum(shadow_hand_dof_vel ** 2, dim=-1)
    joint_vel_reward = -1e-8 * joint_vel_penalty

    # --- CRITICAL: Stronger penalty for dropped/frozen object ---
    angvel_norm = torch.norm(object_angvel, dim=-1)
    # If object is not spinning AND orientation is bad → likely dropped
    frozen_penalty = torch.where(
        (angvel_norm < 1.0) & (rot_dist > 1.0),
        -2.0,
        torch.zeros_like(rot_dist)
    )
    # Additional fall-like penalty if object is tumbling randomly (high angvel but bad orientation)
    tumble_penalty = torch.where(
        (angvel_norm > 8.0) & (rot_dist > 1.5),
        -1.0,
        torch.zeros_like(rot_dist)
    )

    total_reward = rot_reward + spin_reward + action_reward + joint_vel_reward + frozen_penalty + tumble_penalty

    reward_components = {
        "rot_reward": rot_reward,
        "spin_reward": spin_reward,
        "action_reward": action_reward,
        "joint_vel_reward": joint_vel_reward,
        "frozen_penalty": frozen_penalty,
        "tumble_penalty": tumble_penalty
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