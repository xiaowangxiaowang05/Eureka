def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    shadow_hand_dof_vel: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute relative rotation quaternion (object to goal)
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    
    # Orientation error as angle (0 to pi)
    w = quat_diff[:, 3]
    w = torch.clamp(w, -1.0, 1.0)
    rot_angle_error = 2.0 * torch.acos(torch.abs(w))
    
    # Dense orientation reward
    rot_temp = 1.0
    rot_reward = torch.exp(-rot_temp * rot_angle_error)
    
    # Desired spin axis from minimal rotation
    axis_scale = torch.norm(quat_diff[:, :3], dim=-1).unsqueeze(-1)
    spin_axis_desired = torch.where(axis_scale > 1e-6, quat_diff[:, :3] / axis_scale, torch.zeros_like(quat_diff[:, :3]))
    
    no_rotation = (axis_scale.squeeze(-1) < 1e-6)
    
    # Project angular velocity onto desired axis
    angvel_on_axis = torch.sum(object_angvel * spin_axis_desired, dim=-1)
    angvel_off_axis = torch.norm(object_angvel - angvel_on_axis.unsqueeze(-1) * spin_axis_desired, dim=-1)
    
    # Reward correct-axis spinning with target magnitude
    target_angvel_mag = 5.0
    angvel_mag_error = torch.abs(torch.abs(angvel_on_axis) - target_angvel_mag)
    angvel_axis_reward = torch.exp(-0.2 * angvel_mag_error)
    
    # Penalize off-axis motion
    off_axis_penalty = torch.exp(-1.5 * angvel_off_axis)
    
    spin_reward = angvel_axis_reward * off_axis_penalty
    spin_reward = torch.where(no_rotation, torch.exp(-2.0 * torch.norm(object_angvel, dim=-1)), spin_reward)
    
    # Action regularization (very light)
    action_penalty = torch.sum(actions ** 2, dim=-1)
    action_reward = -0.0002 * action_penalty

    # Joint velocity penalty (minimal)
    joint_vel_norm = torch.sum(shadow_hand_dof_vel ** 2, dim=-1)
    joint_vel_reward = -1e-7 * joint_vel_norm

    # Grasp stability: assume object position is near hand (z ~ 0.5), penalize large vertical drop
    # Since we don't have object_pos in inputs, we infer instability from high angular velocity without alignment
    # Instead, we note that dropping leads to erratic angvel → already penalized by spin_reward when not aligned
    # But critical: if object is dropped, angvel becomes noisy and spin_reward drops → implicit penalty
    
    # However, to explicitly prevent dropping, we must have object_pos. But per function signature, it's not available.
    # Given constraints, we rely on spin_reward collapse when object is lost (angvel becomes random)
    
    total_reward = rot_reward + spin_reward + action_reward + joint_vel_reward

    reward_components = {
        "rot_reward": rot_reward,
        "spin_reward": spin_reward,
        "action_reward": action_reward,
        "joint_vel_reward": joint_vel_reward
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