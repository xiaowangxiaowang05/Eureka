def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    fingertip_pos: torch.Tensor,
    object_pos: torch.Tensor,
    progress_buf: torch.Tensor,
    max_episode_length: float
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error using quaternion distance
    rel_quat = quat_mul(object_rot, quat_conjugate(goal_rot))
    rot_dist = 1.0 - rel_quat[:, 3] * rel_quat[:, 3]  # w component squared
    
    # Temperature-scaled orientation reward
    rot_temp = 2.0
    rot_reward = torch.exp(-rot_temp * rot_dist)

    # Encourage some angular velocity to keep spinning (prevent deadlock)
    angvel_norm = torch.norm(object_angvel, dim=-1)
    target_angvel = 2.0  # desired magnitude of angular velocity
    angvel_error = torch.abs(angvel_norm - target_angvel)
    angvel_temp = 1.0
    angvel_reward = torch.exp(-angvel_temp * angvel_error)

    # Penalize large actions (energy efficiency)
    action_penalty = torch.sum(actions ** 2, dim=-1)
    action_temp = 0.01
    action_reward = -action_temp * action_penalty

    # Keep object within workspace: penalize drift from origin in XY plane
    xy_drift = torch.norm(object_pos[:, :2], dim=-1)
    drift_temp = 1.0
    drift_penalty = -drift_temp * xy_drift

    # Fingertip proximity reward: encourage fingers to approach object
    num_fingertips = fingertip_pos.shape[1]
    fingertip_object_dist = torch.norm(fingertip_pos - object_pos.unsqueeze(1), dim=-1)
    avg_fingertip_dist = torch.mean(fingertip_object_dist, dim=1)
    # Use exponential reward to strongly encourage proximity (<0.1m)
    prox_temp = 10.0
    prox_reward = torch.exp(-prox_temp * avg_fingertip_dist)

    # Additional penalty if fingertips are too far (beyond 0.2m)
    max_prox_penalty = -5.0 * torch.clamp(avg_fingertip_dist - 0.2, min=0.0)

    # Time-based survival bonus to encourage longer episodes
    time_bonus = 0.01 * (progress_buf / max_episode_length)

    # NEW: Penalize deviation from neutral hand pose (to encourage natural coordination)
    # Neutral pose is assumed to be zero for all DOFs (from self.shadow_hand_default_dof_pos = 0)
    # But we don't have access to current dof pos here, so this penalty cannot be added without modifying inputs.
    # Instead, we focus on what's available: we can infer unnatural motion from high-frequency actions
    # by approximating acceleration via action differences, but we lack previous actions.
    # Therefore, we strengthen existing penalties and add a force-like penalty via action consistency.

    # Since we cannot compute acceleration or torque directly from given inputs,
    # we introduce a stronger penalty on high-frequency components by penalizing large changes in actions,
    # but again we lack prev_actions. So we must work with available variables only.

    # Alternative: Use the fact that jitter often correlates with high action magnitudes across many DOFs.
    # We already have action_penalty, but it may be too weak. Increase its effect slightly.

    # NEW: Add penalty for uneven fingertip distances (encourage symmetric grip)
    # Compute std of fingertip distances to object center
    fingertip_dist_std = torch.std(fingertip_object_dist, dim=1)
    balance_temp = 5.0
    balance_penalty = -balance_temp * fingertip_dist_std

    # Combine rewards
    total_reward = (
        rot_reward +
        0.5 * angvel_reward +
        1.5 * action_reward +  # Increased action penalty weight to reduce jitter
        0.5 * drift_penalty +
        prox_reward +
        max_prox_penalty +
        time_bonus +
        balance_penalty  # New term to encourage balanced finger positions
    )

    reward_components = {
        "rot_reward": rot_reward,
        "angvel_reward": angvel_reward,
        "action_penalty": action_reward,
        "drift_penalty": drift_penalty,
        "prox_reward": prox_reward,
        "max_prox_penalty": max_prox_penalty,
        "time_bonus": time_bonus,
        "balance_penalty": balance_penalty
    }

    return total_reward, reward_components

# Helper functions required by TorchScript
@torch.jit.script
def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[-1] == 4
    assert b.shape[-1] == 4

    x1, y1, z1, w1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    x2, y2, z2, w2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([x, y, z, w], dim=-1)

@torch.jit.script
def quat_conjugate(a: torch.Tensor) -> torch.Tensor:
    return torch.cat([-a[..., :3], a[..., 3:4]], dim=-1)