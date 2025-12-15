def compute_reward(
    object_pos: torch.Tensor,
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    fingertip_pos: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute relative rotation quaternion (object to goal)
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    
    # Orientation error as angle (0 to pi)
    w = quat_diff[:, 3]
    w = torch.clamp(w, -1.0, 1.0)
    rot_angle_error = 2.0 * torch.acos(torch.abs(w))
    
    # Dense orientation reward
    rot_temp = 2.0
    rot_reward = torch.exp(-rot_temp * rot_angle_error)
    
    # Determine the minimal rotation axis (only if rotation is needed)
    sin_half_angle = torch.norm(quat_diff[:, :3], dim=-1)
    cos_half_angle = torch.abs(quat_diff[:, 3])
    small_rotation = cos_half_angle > 0.9999  # ~ <1 degree
    
    # Avoid division by zero; use identity axis when no rotation
    rotation_axis = torch.where(small_rotation.unsqueeze(-1), 
                                torch.zeros_like(quat_diff[:, :3]),
                                quat_diff[:, :3] / torch.clamp(sin_half_angle.unsqueeze(-1), min=1e-6))
    
    # Project angular velocity onto desired rotation axis
    angvel_projection = torch.sum(object_angvel * rotation_axis, dim=-1)
    
    # Desired spinning direction: should match sign of rotation (via quat_diff w component doesn't give direction,
    # but quat_diff vector part does — positive projection means spinning toward goal)
    # We want magnitude AND correct direction
    target_angvel = 4.0
    angvel_error = torch.abs(angvel_projection - target_angvel)
    
    # Strongly reward correct-direction spinning
    spin_temp = 1.5
    spin_reward = torch.exp(-spin_temp * angvel_error)
    
    # If already aligned (small rotation), discourage any spinning
    spin_reward = torch.where(small_rotation, torch.exp(-3.0 * torch.norm(object_angvel, dim=-1)), spin_reward)
    
    # Contact reward: all fingertips close to object
    fingertip_object_dist = torch.norm(fingertip_pos - object_pos.unsqueeze(1), dim=-1)
    mean_fingertip_dist = torch.mean(fingertip_object_dist, dim=-1)
    contact_temp = 15.0
    contact_reward = torch.exp(-contact_temp * mean_fingertip_dist)
    
    # Light action penalty
    action_penalty = torch.sum(actions ** 2, dim=-1)
    action_reward = -0.005 * action_penalty

    total_reward = 2.0 * rot_reward + 3.0 * spin_reward + 1.0 * contact_reward + action_reward

    reward_components = {
        "rot_reward": rot_reward,
        "spin_reward": spin_reward,
        "contact_reward": contact_reward,
        "action_reward": action_reward,
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