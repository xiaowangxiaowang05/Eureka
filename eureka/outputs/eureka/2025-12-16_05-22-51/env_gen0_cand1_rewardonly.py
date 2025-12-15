def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    shadow_hand_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error using quaternion distance
    # quat_mul(q1, quat_conjugate(q2)) gives the relative rotation from q2 to q1
    rel_quat = quat_mul(object_rot, quat_conjugate(goal_rot))
    # The norm of the vector part of the relative quaternion relates to angular error
    rot_error = torch.norm(rel_quat[:, 1:], dim=1)  # Ignore scalar (w) component
    
    # Temperature for orientation reward shaping
    rot_temp = 1.0
    rot_reward = torch.exp(-rot_temp * rot_error)
    
    # Encourage appropriate angular velocity magnitude for spinning
    # Target angular velocity magnitude (heuristic)
    target_angvel_mag = 2.0
    angvel_mag = torch.norm(object_angvel, dim=1)
    angvel_error = torch.abs(angvel_mag - target_angvel_mag)
    angvel_temp = 0.5
    angvel_reward = torch.exp(-angvel_temp * angvel_error)
    
    # Regularization penalties
    action_norm = torch.norm(actions, dim=1)
    action_penalty = -0.01 * action_norm
    
    joint_vel_norm = torch.norm(shadow_hand_dof_vel, dim=1)
    joint_vel_penalty = -0.001 * joint_vel_norm
    
    # Combine rewards
    total_reward = (
        rot_reward +
        0.5 * angvel_reward +
        action_penalty +
        joint_vel_penalty
    )
    
    reward_components = {
        "rot_reward": rot_reward,
        "angvel_reward": angvel_reward,
        "action_penalty": action_penalty,
        "joint_vel_penalty": joint_vel_penalty
    }
    
    return total_reward, reward_components


# Helper functions needed for TorchScript compatibility
@torch.jit.script
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([-q[:, :3], q[:, 3:4]], dim=-1)

@torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1[:, 3:4], q1[:, 0:1], q1[:, 1:2], q1[:, 2:3]
    w2, x2, y2, z2 = q2[:, 3:4], q2[:, 0:1], q2[:, 1:2], q2[:, 2:3]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return torch.cat([x, y, z, w], dim=-1)