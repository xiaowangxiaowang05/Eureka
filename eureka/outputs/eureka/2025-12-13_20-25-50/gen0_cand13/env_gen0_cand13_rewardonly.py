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
    # Quaternion error as angle (magnitude of rotation vector)
    # Using the formula: theta = 2 * acos(|w|), but we use norm of imaginary part for smoothness
    rot_error = torch.norm(rel_quat[:, 1:], dim=-1)  # norm of [x, y, z] part
    # Alternative: use 1 - |w| which is also smooth and bounded
    # But norm of imaginary part works well and is differentiable everywhere except identity
    
    # Temperature parameter for orientation reward shaping
    orient_temp = 2.0
    orient_reward = torch.exp(-orient_temp * rot_error)
    
    # Angular velocity regularization: encourage spinning if needed, but avoid excessive spin
    # For pure orientation task without specified rotation axis/speed, we penalize high angular velocity
    angvel_norm = torch.norm(object_angvel, dim=-1)
    angvel_temp = 0.5
    angvel_penalty = torch.exp(-angvel_temp * angvel_norm)
    
    # Action regularization: minimize control effort
    action_penalty = torch.sum(actions ** 2, dim=-1)
    action_temp = 0.01
    action_reward = torch.exp(-action_temp * action_penalty)
    
    # Joint velocity penalty to reduce jitter
    joint_vel_norm = torch.sum(shadow_hand_dof_vel ** 2, dim=-1)
    joint_vel_temp = 0.001
    joint_vel_reward = torch.exp(-joint_vel_temp * joint_vel_norm)
    
    # Combine rewards with appropriate weights
    total_reward = (
        1.0 * orient_reward +
        0.3 * angvel_penalty +
        0.2 * action_reward +
        0.1 * joint_vel_reward
    )
    
    reward_components = {
        "orient_reward": orient_reward,
        "angvel_penalty": angvel_penalty,
        "action_reward": action_reward,
        "joint_vel_reward": joint_vel_reward
    }
    
    return total_reward, reward_components

# Helper functions required for torch.jit.script compatibility
@torch.jit.script
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([-q[:, :3], q[:, 3:4]], dim=-1)

@torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return torch.stack([x, y, z, w], dim=-1)