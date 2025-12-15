def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    progress_buf: torch.Tensor,
    max_episode_length: float
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute rotation error as angle between object and goal orientations
    rel_quat = quat_mul(object_rot, quat_conjugate(goal_rot))
    rot_error = 2.0 * torch.asin(torch.clamp(torch.norm(rel_quat[:, 0:3], p=2, dim=-1), max=1.0))  # [0, pi]
    
    # Dense orientation reward: higher when closer to target
    orient_reward = torch.exp(-2.0 * rot_error)
    
    # Small penalty for unnecessary angular velocity (only when not near target)
    angvel_norm = torch.norm(object_angvel, dim=-1)
    angvel_penalty = torch.where(rot_error < 0.2, 0.05 * angvel_norm, 0.01 * angvel_norm)
    
    # Very light action regularization to avoid excessive energy use
    action_penalty = 0.001 * torch.sum(actions ** 2, dim=-1)
    
    # Bonus for being very close to target orientation
    success_bonus = torch.where(rot_error < 0.05, 2.0, 0.0)
    
    # Minor time penalty to encourage efficiency
    time_penalty = 0.01 * (progress_buf / max_episode_length)
    
    total_reward = orient_reward - angvel_penalty - action_penalty + success_bonus - time_penalty
    
    reward_components = {
        "orient_reward": orient_reward,
        "angvel_penalty": -angvel_penalty,
        "action_penalty": -action_penalty,
        "success_bonus": success_bonus,
        "time_penalty": -time_penalty
    }
    
    return total_reward, reward_components

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