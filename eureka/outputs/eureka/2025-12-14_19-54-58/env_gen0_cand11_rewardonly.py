def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    progress_buf: torch.Tensor,
    max_episode_length: int
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error using quaternion distance
    # quat_mul(q1, quat_conjugate(q2)) gives relative rotation from q2 to q1
    rel_quat = torch.mul(object_rot, quat_conjugate(goal_rot))
    # The w component of the relative quaternion relates to the angle difference
    # Orientation error: 1 - |w|, since w = cos(theta/2), so smaller angle => larger |w|
    orient_err = 1.0 - torch.abs(rel_quat[:, 3])  # shape: [num_envs]
    
    # Angular velocity magnitude (penalize excessive spinning beyond what's needed)
    angvel_norm = torch.norm(object_angvel, dim=-1)
    
    # Action regularization (smoothness and energy efficiency)
    action_norm = torch.norm(actions, dim=-1)
    
    # Reward components with temperature scaling for proper shaping
    orient_temp = 2.0
    angvel_temp = 0.5
    action_temp = 0.01
    
    orient_reward = torch.exp(-orient_temp * orient_err)
    angvel_penalty = torch.exp(-angvel_temp * angvel_norm)
    action_penalty = torch.exp(-action_temp * action_norm)
    
    # Total reward combines orientation alignment and controlled motion
    total_reward = orient_reward * angvel_penalty + 0.1 * action_penalty
    
    reward_components = {
        "orient_reward": orient_reward,
        "angvel_penalty": angvel_penalty,
        "action_penalty": action_penalty
    }
    
    return total_reward, reward_components


# Helper functions required for torch.jit compatibility
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