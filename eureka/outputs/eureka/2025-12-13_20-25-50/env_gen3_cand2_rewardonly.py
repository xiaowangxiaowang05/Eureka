def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    shadow_hand_dof_vel: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute relative rotation and extract angle-axis representation
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    # Ensure w in [-1, 1]
    w = torch.clamp(quat_diff[:, 3], -1.0, 1.0)
    rot_angle = 2.0 * torch.acos(torch.abs(w))  # [0, pi]
    
    # Dense orientation reward: higher when closer to target
    rot_reward = torch.exp(-2.0 * rot_angle)
    
    # Extract rotation axis (unit vector); handle zero-rotation case
    sin_half_theta = torch.sqrt(1.0 - w * w)
    axis = torch.where(sin_half_theta.unsqueeze(-1) > 1e-6, 
                       quat_diff[:, :3] / sin_half_theta.unsqueeze(-1), 
                       torch.zeros_like(quat_diff[:, :3]))
    
    # Desired angular velocity direction: along shortest rotation axis
    # Magnitude scaled by remaining angle to encourage faster spin when far
    desired_angvel = axis * (rot_angle.unsqueeze(-1))  # magnitude proportional to error
    
    # Angular velocity tracking reward
    angvel_error = torch.norm(object_angvel - desired_angvel, dim=-1)
    angvel_reward = torch.exp(-0.5 * angvel_error)
    
    # Action regularization (very light)
    action_penalty = torch.sum(actions ** 2, dim=-1)
    action_reward = -0.0005 * action_penalty

    # Joint velocity penalty (light)
    joint_vel_reward = -1e-6 * torch.sum(shadow_hand_dof_vel ** 2, dim=-1)
    
    # Critical: detect dropped/frozen object via low angular velocity AND high orientation error
    # This acts as a proxy for drop since object_pos is not available
    frozen = (torch.norm(object_angvel, dim=-1) < 0.7) & (rot_angle > 0.8)
    drop_penalty = torch.where(frozen, -3.0, torch.zeros_like(rot_angle))
    
    total_reward = rot_reward + angvel_reward + action_reward + joint_vel_reward + drop_penalty

    reward_components = {
        "rot_reward": rot_reward,
        "angvel_reward": angvel_reward,
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
    conj = torch.cat([-a[..., 0:3], a[..., 3:4]], dim=-1)
    return conj