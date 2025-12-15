def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    shadow_hand_dof_vel: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error using quaternion distance
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    rot_dist = 1.0 - quat_diff[:, 3]  # [0, 2]

    rot_temp = 2.0
    rot_reward = torch.exp(-rot_temp * rot_dist)

    # Direction-aware spinning: project angular velocity onto the axis that reduces orientation error
    # Get rotation vector (axis-angle representation) from relative quaternion
    angle = 2.0 * torch.acos(torch.clamp(quat_diff[:, 3], -1.0, 1.0))  # [0, pi]
    sin_half_angle = torch.sqrt(1.0 - quat_diff[:, 3] * quat_diff[:, 3])
    axis = torch.where(sin_half_angle.unsqueeze(-1) > 1e-6, 
                       quat_diff[:, :3] / sin_half_angle.unsqueeze(-1), 
                       torch.zeros_like(quat_diff[:, :3]))
    
    # Desired angular velocity direction is along the error axis
    desired_angvel_dir = axis
    
    # Project actual angular velocity onto desired direction
    angvel_along_axis = torch.sum(object_angvel * desired_angvel_dir, dim=-1)
    
    # Encourage positive spin in the correct direction (magnitude adaptive to remaining error)
    target_angvel_mag = torch.clamp(angle * 5.0, min=0.5, max=10.0)  # Scale with remaining rotation needed
    angvel_error = torch.abs(angvel_along_axis - target_angvel_mag)
    
    angvel_temp = 0.5
    angvel_reward = torch.exp(-angvel_temp * angvel_error)

    # Action regularization
    action_penalty = torch.sum(actions ** 2, dim=-1)
    action_temp = 0.01
    action_reward = torch.exp(-action_temp * action_penalty)

    # Joint velocity smoothness
    joint_vel_penalty = torch.sum(shadow_hand_dof_vel ** 2, dim=-1)
    joint_vel_temp = 0.001
    joint_vel_reward = torch.exp(-joint_vel_temp * joint_vel_penalty)

    total_reward = (
        2.0 * rot_reward +
        1.5 * angvel_reward +   # Increased weight to drive active spinning
        0.1 * action_reward +
        0.1 * joint_vel_reward
    )

    reward_components = {
        "rot_reward": rot_reward,
        "angvel_reward": angvel_reward,
        "action_reward": action_reward,
        "joint_vel_reward": joint_vel_reward
    }

    return total_reward, reward_components

# Helper functions required for TorchScript compatibility
@torch.jit.script
def quat_conjugate(q):
    return torch.cat([-q[:, :3], q[:, 3:4]], dim=-1)

@torch.jit.script
def quat_mul(q1, q2):
    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([x, y, z, w], dim=-1)