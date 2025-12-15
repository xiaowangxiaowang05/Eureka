def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    shadow_hand_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error using quaternion distance
    # quat_mul(q1, quat_conjugate(q2)) gives relative rotation from q2 to q1
    rel_rot = quat_mul(object_rot, quat_conjugate(goal_rot))
    # Extract angle-axis representation; angle is 2*acos(w)
    w = rel_rot[:, 3]  # scalar part of quaternion
    # Clamp w to [-1, 1] to avoid NaNs due to numerical errors
    w = torch.clamp(w, -1.0, 1.0)
    orientation_error = 2.0 * torch.acos(torch.abs(w))  # in [0, pi]
    
    # Temperature for orientation reward shaping
    orientation_temp = 1.0
    orientation_reward = torch.exp(-orientation_temp * orientation_error)
    
    # Angular velocity regularization: encourage smooth rotation
    angvel_norm = torch.norm(object_angvel, dim=-1)
    angvel_temp = 0.1
    angvel_reward = torch.exp(-angvel_temp * angvel_norm)
    
    # Action regularization to reduce jitter and energy use
    action_norm = torch.norm(actions, dim=-1)
    action_temp = 0.05
    action_penalty = -action_temp * action_norm
    
    # Joint velocity penalty to prevent excessive motor speeds
    joint_vel_norm = torch.norm(shadow_hand_dof_vel, dim=-1)
    joint_vel_temp = 0.01
    joint_vel_penalty = -joint_vel_temp * joint_vel_norm
    
    # Combine rewards
    total_reward = (
        orientation_reward 
        + 0.3 * angvel_reward 
        + action_penalty 
        + joint_vel_penalty
    )
    
    reward_components = {
        "orientation_reward": orientation_reward,
        "angvel_reward": angvel_reward,
        "action_penalty": action_penalty,
        "joint_vel_penalty": joint_vel_penalty
    }
    
    return total_reward, reward_components

# Helper functions required for TorchScript compatibility
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