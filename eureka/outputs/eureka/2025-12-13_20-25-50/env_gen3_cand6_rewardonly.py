def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    shadow_hand_dof_vel: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # --- Orientation alignment reward ---
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, :3], dim=-1), max=1.0))
    rot_reward = torch.exp(-2.0 * rot_dist)

    # --- Spin reward (only meaningful if object is aligned) ---
    # Desired spin axis from rotation vector
    axis_norm = torch.norm(quat_diff[:, :3], dim=-1, keepdim=True)
    spin_axis = torch.where(axis_norm > 1e-6, quat_diff[:, :3] / axis_norm, torch.zeros_like(quat_diff[:, :3]))
    
    # Project angular velocity onto desired axis
    angvel_on_axis = torch.sum(object_angvel * spin_axis, dim=-1)
    angvel_mag = torch.abs(angvel_on_axis)
    
    # Encourage moderate spin (not too slow, not too fast) only when close to target orientation
    target_spin = 4.0
    spin_error = torch.abs(angvel_mag - target_spin)
    spin_reward_raw = torch.exp(-0.3 * spin_error)
    
    # Gate spin reward: only active when orientation is reasonably good
    spin_reward = torch.where(rot_dist < 1.0, spin_reward_raw, torch.zeros_like(spin_reward_raw))

    # --- Action & joint velocity penalties (very light) ---
    action_penalty = torch.sum(actions ** 2, dim=-1)
    action_reward = -5e-5 * action_penalty

    joint_vel_penalty = torch.sum(shadow_hand_dof_vel ** 2, dim=-1)
    joint_vel_reward = -1e-7 * joint_vel_penalty

    # --- CRITICAL: Add implicit "hold object" signal via frozen state penalty ---
    # Penalize low angular velocity AND large orientation error (typical of dropped/frozen object)
    angvel_norm = torch.norm(object_angvel, dim=-1)
    frozen_object_penalty = torch.where(
        (angvel_norm < 0.8) & (rot_dist > 0.8),
        -1.5,
        torch.zeros_like(rot_dist)
    )

    # Total reward composition
    total_reward = rot_reward + spin_reward + action_reward + joint_vel_reward + frozen_object_penalty

    reward_components = {
        "rot_reward": rot_reward,
        "spin_reward": spin_reward,
        "action_reward": action_reward,
        "joint_vel_reward": joint_vel_reward,
        "frozen_object_penalty": frozen_object_penalty
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