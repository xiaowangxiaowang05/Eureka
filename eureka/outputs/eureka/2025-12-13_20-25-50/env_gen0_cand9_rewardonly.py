def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    shadow_hand_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error using quaternion distance
    # The rotation error is measured as the angle of the relative rotation
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    rot_error = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:], dim=-1), min=0.0, max=1.0))
    
    # Temperature parameter for orientation reward shaping
    rot_temp = 1.0
    rot_reward = torch.exp(-rot_temp * rot_error)
    
    # Angular velocity regularization: encourage some spinning but not too much
    angvel_norm = torch.norm(object_angvel, dim=-1)
    angvel_target = 2.0  # target angular speed in rad/s
    angvel_error = torch.abs(angvel_norm - angvel_target)
    angvel_temp = 0.5
    angvel_reward = torch.exp(-angvel_temp * angvel_error)
    
    # Action regularization to prevent high-frequency jitter
    action_penalty = torch.sum(actions ** 2, dim=-1)
    action_temp = 0.01
    action_reward = -action_temp * action_penalty
    
    # Joint velocity regularization to avoid excessive joint speeds
    joint_vel_penalty = torch.sum(shadow_hand_dof_vel ** 2, dim=-1)
    joint_vel_temp = 0.001
    joint_vel_reward = -joint_vel_temp * joint_vel_penalty
    
    # Combine rewards
    total_reward = (
        rot_reward +
        0.5 * angvel_reward +
        action_reward +
        joint_vel_reward
    )
    
    reward_components = {
        "rot_reward": rot_reward,
        "angvel_reward": angvel_reward,
        "action_penalty": action_reward,
        "joint_vel_penalty": joint_vel_reward
    }
    
    return total_reward, reward_components

# Helper functions required for quaternion operations
@torch.jit.script
def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape == b.shape
    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return torch.stack([x, y, z, w], dim=-1)

@torch.jit.script
def quat_conjugate(a: torch.Tensor) -> torch.Tensor:
    return torch.cat([-a[:, :3], a[:, 3:4]], dim=-1)