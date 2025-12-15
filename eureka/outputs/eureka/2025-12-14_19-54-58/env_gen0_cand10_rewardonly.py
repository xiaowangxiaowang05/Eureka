def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    progress_buf: torch.Tensor,
    max_episode_length: float
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error using quaternion distance
    # quat_mul(object_rot, quat_conjugate(goal_rot)) gives relative rotation
    # The scalar part (w) of the resulting quaternion indicates alignment
    rel_quat = quat_mul(object_rot, quat_conjugate(goal_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(rel_quat[:, 1:], dim=-1), min=0.0, max=1.0))  # in [0, pi]
    
    # Orientation reward: encourage alignment with target orientation
    rot_reward_temp = 1.0
    rot_reward = torch.exp(-rot_reward_temp * rot_dist)
    
    # Angular velocity regularization: discourage excessive spinning once aligned
    angvel_norm = torch.norm(object_angvel, dim=-1)
    angvel_penalty_temp = 0.1
    angvel_penalty = torch.exp(-angvel_penalty_temp * angvel_norm)
    
    # Only apply angvel penalty when close to goal orientation to avoid conflicting incentives
    angvel_penalty_weight = 0.5
    combined_rot_angvel_reward = rot_reward + angvel_penalty_weight * rot_reward * (1.0 - angvel_penalty)
    
    # Action regularization: penalize large actions for energy efficiency
    action_penalty_temp = 0.05
    action_penalty = torch.sum(actions ** 2, dim=-1)
    action_reward = torch.exp(-action_penalty_temp * action_penalty)
    
    # Time-based survival bonus to encourage longer episodes (helps exploration)
    survival_bonus = 0.01 * torch.ones_like(progress_buf, dtype=torch.float32)
    
    # Total reward composition
    total_reward = (
        2.0 * combined_rot_angvel_reward +
        0.5 * action_reward +
        survival_bonus
    )
    
    reward_components = {
        "rot_reward": rot_reward,
        "angvel_penalty": angvel_penalty,
        "action_reward": action_reward,
        "survival_bonus": survival_bonus,
        "combined_rot_angvel_reward": combined_rot_angvel_reward
    }
    
    return total_reward, reward_components

# Helper functions required for TorchScript compatibility
@torch.jit.script
def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[-1] == 4
    assert b.shape[-1] == 4
    
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    
    return torch.stack([w, x, y, z], dim=-1)

@torch.jit.script
def quat_conjugate(a: torch.Tensor) -> torch.Tensor:
    return torch.cat([-a[..., :3], a[..., 3:]], dim=-1)