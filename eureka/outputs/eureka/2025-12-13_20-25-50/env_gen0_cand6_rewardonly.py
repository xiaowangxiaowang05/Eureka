def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    progress_buf: torch.Tensor,
    max_episode_length: float
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error using quaternion distance
    # quat_mul(q1, quat_conjugate(q2)) gives relative rotation from q2 to q1
    rel_quat = quat_mul(object_rot, quat_conjugate(goal_rot))
    # The scalar part (w component) of the relative quaternion indicates alignment
    # When object_rot == goal_rot, rel_quat = [0,0,0,1] (identity), so w=1
    orientation_error = 1.0 - rel_quat[:, 3]  # 1 - cos(theta/2), ranges [0, 2]
    
    # Angular velocity regularization: penalize excessive spinning when close to target
    angvel_norm = torch.norm(object_angvel, dim=-1)
    angvel_penalty = torch.where(orientation_error < 0.1, angvel_norm, torch.zeros_like(angvel_norm))
    
    # Action regularization to encourage energy efficiency
    action_penalty = torch.sum(actions ** 2, dim=-1)
    
    # Success bonus for being very close to target orientation
    success_bonus = torch.where(orientation_error < 0.05, 5.0, 0.0)
    
    # Time-based penalty to encourage faster completion (optional but often helpful)
    time_penalty = progress_buf / max_episode_length * 0.1
    
    # Temperature parameters for reward shaping
    orient_temp = 2.0
    angvel_temp = 0.1
    action_temp = 0.01
    
    # Shaped rewards using exponential transformation for smooth gradients
    orient_reward = torch.exp(-orient_temp * orientation_error)
    angvel_reward = -angvel_temp * angvel_penalty
    action_reward = -action_temp * action_penalty
    
    total_reward = orient_reward + angvel_reward + action_reward + success_bonus - time_penalty
    
    reward_components = {
        "orient_reward": orient_reward,
        "angvel_reward": angvel_reward,
        "action_reward": action_reward,
        "success_bonus": success_bonus,
        "time_penalty": -time_penalty
    }
    
    return total_reward, reward_components

# Helper functions required for quaternion operations
@torch.jit.script
def quat_conjugate(q):
    return torch.cat([-q[:, :3], q[:, 3:4]], dim=-1)

@torch.jit.script
def quat_mul(q1, q2):
    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return torch.stack([x, y, z, w], dim=-1)