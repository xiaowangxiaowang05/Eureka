def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    shadow_hand_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error using quaternion distance
    # quat_mul(q1, quat_conjugate(q2)) gives relative rotation from q2 to q1
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    # Extract angle from quaternion (angle = 2 * acos(|w|), but we use 1 - |w| as proxy)
    rot_error = 1.0 - torch.abs(quat_diff[:, 3])  # w component is last in [x,y,z,w]
    
    # Temperature parameter for orientation reward shaping
    rot_temp = 1.0
    rot_reward = torch.exp(-rot_temp * rot_error)
    
    # Angular velocity reward: encourage some spin but not too much chaotic motion
    # Target angular velocity magnitude - if task requires sustained spinning
    angvel_norm = torch.norm(object_angvel, dim=-1)
    target_angvel_mag = 2.0  # reasonable target spin rate
    angvel_error = torch.abs(angvel_norm - target_angvel_mag)
    angvel_temp = 0.5
    angvel_reward = torch.exp(-angvel_temp * angvel_error)
    
    # Action regularization: penalize large actions
    action_penalty = torch.sum(actions ** 2, dim=-1)
    action_reward = -0.01 * action_penalty
    
    # Joint velocity regularization: prevent overly fast joint movements
    joint_vel_penalty = torch.sum(shadow_hand_dof_vel ** 2, dim=-1)
    joint_vel_reward = -0.001 * joint_vel_penalty
    
    # Combine rewards
    total_reward = rot_reward + 0.5 * angvel_reward + action_reward + joint_vel_reward
    
    reward_components = {
        "rot_reward": rot_reward,
        "angvel_reward": angvel_reward,
        "action_reward": action_reward,
        "joint_vel_reward": joint_vel_reward
    }
    
    return total_reward, reward_components

# Helper functions needed for TorchScript compatibility
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