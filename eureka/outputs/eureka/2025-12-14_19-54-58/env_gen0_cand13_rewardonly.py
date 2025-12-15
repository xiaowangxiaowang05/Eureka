def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    dof_vel: torch.Tensor,
    progress_buf: torch.Tensor,
    max_episode_length: float
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error using quaternion distance
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    angle_error = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], dim=-1), min=0.0, max=1.0))
    
    # Reward for reducing orientation error (higher when closer to target)
    orientation_reward_temp = 1.0
    orientation_reward = torch.exp(-orientation_reward_temp * angle_error)
    
    # Encourage maintaining some angular velocity to keep spinning (avoid static solution)
    angvel_norm = torch.norm(object_angvel, dim=-1)
    desired_angvel = 2.0  # target angular speed magnitude
    angvel_error = torch.abs(angvel_norm - desired_angvel)
    angvel_reward_temp = 0.5
    angvel_reward = torch.exp(-angvel_reward_temp * angvel_error)
    
    # Action regularization to reduce jitter and energy use
    action_penalty = torch.sum(actions ** 2, dim=-1)
    action_reward = -0.01 * action_penalty
    
    # Joint velocity penalty for smooth motion
    joint_vel_penalty = torch.sum(dof_vel ** 2, dim=-1)
    joint_vel_reward = -0.001 * joint_vel_penalty
    
    # Time-based bonus to encourage faster completion
    time_bonus = 0.01 * (1.0 - (progress_buf / max_episode_length))
    
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

# Helper functions required for torch.jit.script compatibility
@torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)

@torch.jit.script
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[:, 0:1], -q[:, 1:4]], dim=-1)