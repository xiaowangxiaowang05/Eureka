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
    
    # Dense orientation reward with appropriate temperature
    rot_temp = 2.0
    rot_reward = torch.exp(-rot_temp * rot_angle_error)
    
    # Compute the axis and magnitude of the shortest rotation from object to goal
    sin_half_theta = torch.norm(quat_diff[:, :3], dim=-1)
    cos_half_theta = torch.abs(w)
    # Avoid division by zero
    half_theta = torch.atan2(sin_half_theta, cos_half_theta)
    theta = 2.0 * half_theta  # total rotation angle [0, pi]
    
    # Unit vector of rotation axis (undefined when theta=0, but we handle it)
    axis = torch.where(sin_half_theta.unsqueeze(-1) > 1e-6, 
                       quat_diff[:, :3] / sin_half_theta.unsqueeze(-1), 
                       torch.zeros_like(quat_diff[:, :3]))
    
    # Desired angular velocity direction: along the rotation axis to reduce error
    # Magnitude scaled by remaining error: more error => need faster spin
    desired_angvel = axis * theta.unsqueeze(-1) * 5.0  # Scale factor for desired speed
    
    # Penalize deviation of actual angvel from desired angvel
    angvel_error = torch.norm(object_angvel - desired_angvel, dim=-1)
    angvel_temp = 0.5
    angvel_reward = torch.exp(-angvel_temp * angvel_error)
    
    # Contact reward: encourage all fingertips to stay close to the object
    fingertip_object_dist = torch.norm(fingertip_pos - object_pos.unsqueeze(1), dim=-1)
    mean_fingertip_dist = torch.mean(fingertip_object_dist, dim=-1)
    contact_temp = 8.0
    contact_reward = torch.exp(-contact_temp * mean_fingertip_dist)
    
    # Light action regularization
    action_penalty = torch.sum(actions ** 2, dim=-1)
    action_reg_weight = 0.001
    action_reward = -action_reg_weight * action_penalty

    total_reward = 2.0 * rot_reward + 1.0 * angvel_reward + 1.5 * contact_reward + action_reward

    reward_components = {
        "rot_reward": rot_reward,
        "angvel_reward": angvel_reward,
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