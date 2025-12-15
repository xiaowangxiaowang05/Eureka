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
    
    # Dense orientation reward with learnable temperature (scaled for better gradient)
    orient_temp: float = 1.5
    orient_reward = torch.exp(-rot_error / orient_temp)
    
    # Light action regularization to avoid jitter but not freeze
    action_penalty = 0.0005 * torch.sum(actions ** 2, dim=-1)
    
    # Small bonus for being very close to target orientation
    success_bonus = torch.where(rot_error < 0.05, 1.0, 0.0)
    
    total_reward = orient_reward - action_penalty + success_bonus
    
    reward_components = {
        "orient_reward": orient_reward,
        "action_penalty": -action_penalty,
        "success_bonus": success_bonus
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