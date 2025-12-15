def compute_reward(
    object_pos: torch.Tensor,
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    shadow_hand_dof_vel: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    w = quat_diff[:, 3]
    w = torch.clamp(w, -1.0, 1.0)
    rot_angle_error = 2.0 * torch.acos(torch.abs(w))
    
    # Dense orientation reward
    rot_reward = torch.exp(-2.0 * rot_angle_error)
    
    # Height maintenance: penalize falling below grasp height (~0.5)
    target_height = 0.5
    height_error = torch.abs(object_pos[:, 2] - target_height)
    height_reward = torch.exp(-5.0 * height_error)
    
    # Spin reward: only if not dropped and aligned
    spin_axis = quat_diff[:, :3]
    axis_norm = torch.norm(spin_axis, dim=-1, keepdim=True)
    valid_axis = axis_norm > 1e-6
    spin_axis = torch.where(valid_axis, spin_axis / axis_norm, torch.zeros_like(spin_axis))
    
    angvel_on_axis = torch.sum(object_angvel * spin_axis, dim=-1)
    desired_spin = 4.0  # rad/s
    spin_magnitude_error = torch.abs(torch.abs(angvel_on_axis) - desired_spin)
    spin_axis_reward = torch.exp(-0.8 * spin_magnitude_error)
    
    off_axis_vel = object_angvel - angvel_on_axis.unsqueeze(-1) * spin_axis
    off_axis_penalty = torch.exp(-2.0 * torch.norm(off_axis_vel, dim=-1))
    
    spin_reward = spin_axis_reward * off_axis_penalty
    # If no rotation needed (aligned), just maintain low spin
    spin_reward = torch.where(axis_norm.squeeze(-1) < 0.1, 
                              torch.exp(-1.0 * torch.norm(object_angvel, dim=-1)), 
                              spin_reward)
    
    # Action regularization
    action_penalty = torch.sum(actions ** 2, dim=-1)
    action_reward = -0.00005 * action_penalty

    # Joint velocity penalty (light)
    joint_vel_penalty = torch.sum(shadow_hand_dof_vel ** 2, dim=-1)
    joint_vel_reward = -5e-8 * joint_vel_penalty

    # Drop penalty: strong negative if too low
    drop_penalty = torch.where(object_pos[:, 2] < 0.2, -3.0, torch.zeros_like(rot_angle_error))
    
    total_reward = (
        rot_reward 
        + height_reward 
        + 0.5 * spin_reward 
        + action_reward 
        + joint_vel_reward 
        + drop_penalty
    )

    reward_components = {
        "rot_reward": rot_reward,
        "height_reward": height_reward,
        "spin_reward": spin_reward,
        "action_reward": action_reward,
        "joint_vel_reward": joint_vel_reward,
        "drop_penalty": drop_penalty
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
    return torch.cat([-a[..., 0:3], a[..., 3:4]], dim=-1)