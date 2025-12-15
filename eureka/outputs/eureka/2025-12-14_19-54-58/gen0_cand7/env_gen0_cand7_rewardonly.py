def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    shadow_hand_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error using quaternion distance
    # quat_mul(q1, quat_conjugate(q2)) gives relative rotation from q2 to q1
    rel_quat = quat_mul(object_rot, quat_conjugate(goal_rot))
    # The w component of the relative quaternion relates to the angle difference
    # Distance in SO(3): 1 - |w|, but we use 1 - w^2 for smoothness
    rot_dist = 1.0 - torch.abs(rel_quat[:, 3])  # w component is at index 3
    
    # Orientation reward with temperature scaling
    rot_temp = 2.0
    rot_reward = torch.exp(-rot_dist * rot_temp)
    
    # Angular velocity regularization: penalize excessive spinning
    angvel_norm = torch.norm(object_angvel, dim=-1)
    angvel_temp = 0.5
    angvel_penalty = torch.exp(-angvel_norm * angvel_temp)
    
    # Action regularization to encourage energy efficiency
    action_norm = torch.norm(actions, dim=-1)
    action_temp = 0.1
    action_penalty = torch.exp(-action_norm * action_temp)
    
    # Joint velocity regularization to prevent jitter
    joint_vel_norm = torch.norm(shadow_hand_dof_vel, dim=-1)
    joint_vel_temp = 0.1
    joint_vel_penalty = torch.exp(-joint_vel_norm * joint_vel_temp)
    
    # Combine rewards with appropriate weights
    total_reward = (
        1.0 * rot_reward +
        0.3 * angvel_penalty +
        0.1 * action_penalty +
        0.1 * joint_vel_penalty
    )
    
    reward_components = {
        "rot_reward": rot_reward,
        "angvel_penalty": angvel_penalty,
        "action_penalty": action_penalty,
        "joint_vel_penalty": joint_vel_penalty
    }
    
    return total_reward, reward_components

# Helper functions required for torch.jit.script compatibility
@torch.jit.script
def quat_conjugate(q):
    return torch.cat((-q[:, :3], q[:, 3:4]), dim=-1)

@torch.jit.script
def quat_mul(q1, q2):
    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return torch.stack([x, y, z, w], dim=-1)