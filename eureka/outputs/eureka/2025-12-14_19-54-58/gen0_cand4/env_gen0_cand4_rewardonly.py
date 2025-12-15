def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    shadow_hand_dof_vel: torch.Tensor,
    progress_buf: torch.Tensor,
    max_episode_length: float
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error using quaternion distance
    # quat_mul(object_rot, quat_conjugate(goal_rot)) gives relative rotation
    rel_quat = quat_mul(object_rot, quat_conjugate(goal_rot))
    # The angle of rotation is 2 * acos(w), but we use 1 - |w| as a proxy for alignment
    # This is differentiable and ranges from 0 (aligned) to 1 (opposite)
    orientation_error = 1.0 - torch.abs(rel_quat[:, 3])  # w component
    
    # Exponential reward for orientation alignment
    orientation_temperature = 2.0
    orientation_reward = torch.exp(-orientation_temperature * orientation_error)
    
    # Encourage smooth spinning by rewarding consistent angular velocity magnitude
    # We don't care about direction since the target is orientation, not continuous spinning
    angvel_magnitude = torch.norm(object_angvel, dim=-1)
    # Penalize very high angular velocities that might be unstable
    angvel_temperature = 0.5
    angvel_reward = torch.exp(-angvel_temperature * torch.abs(angvel_magnitude - 2.0))  # target ~2 rad/s
    
    # Action regularization to encourage energy efficiency
    action_penalty = torch.sum(actions ** 2, dim=-1)
    action_reward = -0.01 * action_penalty
    
    # Joint velocity regularization to prevent jitter
    joint_vel_penalty = torch.sum(shadow_hand_dof_vel ** 2, dim=-1)
    joint_vel_reward = -0.001 * joint_vel_penalty
    
    # Time-based reward shaping: slightly increase reward as episode progresses if doing well
    time_factor = progress_buf / max_episode_length
    time_bonus = 0.1 * time_factor * orientation_reward
    
    # Combine rewards
    total_reward = (
        orientation_reward + 
        0.5 * angvel_reward + 
        action_reward + 
        joint_vel_reward + 
        time_bonus
    )
    
    reward_components = {
        "orientation_reward": orientation_reward,
        "angvel_reward": angvel_reward,
        "action_reward": action_reward,
        "joint_vel_reward": joint_vel_reward,
        "time_bonus": time_bonus
    }
    
    return total_reward, reward_components

# Helper functions for quaternion operations (required for TorchScript)
@torch.jit.script
def quat_conjugate(q):
    return torch.cat([-q[:, :3], q[:, 3:4]], dim=-1)

@torch.jit.script
def quat_mul(q1, q2):
    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * x2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([x, y, z, w], dim=-1)