def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    shadow_hand_dof_vel: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Orientation alignment reward
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    w = quat_diff[:, 3]
    w = torch.clamp(w, -1.0, 1.0)
    rot_angle_error = 2.0 * torch.acos(torch.abs(w))
    rot_reward = torch.exp(-1.5 * rot_angle_error)
    
    # Encourage any motion toward correct orientation without enforcing fixed spin magnitude
    axis_scale = torch.norm(quat_diff[:, :3], dim=-1)
    needs_rotation = axis_scale > 0.05  # Only care about spin if not already aligned
    
    # Desired axis (normalized)
    spin_axis_desired = torch.where(axis_scale.unsqueeze(-1) > 1e-6, 
                                    quat_diff[:, :3] / axis_scale.unsqueeze(-1), 
                                    torch.zeros_like(quat_diff[:, :3]))
    
    # Project angular velocity onto desired axis
    angvel_on_axis = torch.sum(object_angvel * spin_axis_desired, dim=-1)
    angvel_off_axis = torch.norm(object_angvel - angvel_on_axis.unsqueeze(-1) * spin_axis_desired, dim=-1)
    
    # Reward alignment of spin direction (not magnitude)
    direction_alignment = torch.clamp(angvel_on_axis / (torch.norm(object_angvel, dim=-1) + 1e-6), 0.0, 1.0)
    spin_direction_reward = torch.where(needs_rotation, direction_alignment, torch.ones_like(direction_alignment))
    
    # Penalize excessive off-axis spinning only when rotation is needed
    off_axis_penalty = torch.where(needs_rotation, torch.exp(-2.0 * angvel_off_axis), torch.ones_like(angvel_off_axis))
    
    spin_reward = spin_direction_reward * off_axis_penalty

    # Action regularization (very light)
    action_penalty = torch.sum(actions ** 2, dim=-1)
    action_reward = -0.0005 * action_penalty

    # Add implicit stability signal via joint velocity (prevent freezing while allowing motion)
    dof_vel_cost = -0.0001 * torch.sum(shadow_hand_dof_vel ** 2, dim=-1)

    total_reward = rot_reward + 1.0 * spin_reward + action_reward + dof_vel_cost

    reward_components = {
        "rot_reward": rot_reward,
        "spin_reward": spin_reward,
        "action_reward": action_reward,
        "dof_vel_cost": dof_vel_cost,
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