def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    fingertip_pos: torch.Tensor,
    object_pos: torch.Tensor,
    progress_buf: torch.Tensor,
    max_episode_length: float,
    shadow_hand_dof_pos: torch.Tensor,
    shadow_hand_dof_vel: torch.Tensor,
    shadow_hand_dof_default_pos: torch.Tensor,
    vec_sensor_tensor: torch.Tensor,
    dof_force_tensor: torch.Tensor
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

    # NEW: Penalize joint acceleration to reduce jitter
    joint_acc = shadow_hand_dof_vel - torch.roll(shadow_hand_dof_vel, shifts=1, dims=0)
    acc_penalty = -0.0001 * torch.sum(joint_acc ** 2, dim=-1)

    # NEW: Penalize deviation from neutral pose to encourage natural finger posture
    neutral_dev = torch.norm(shadow_hand_dof_pos - shadow_hand_dof_default_pos, dim=-1)
    neutral_penalty = -0.01 * neutral_dev

    # NEW: Penalize excessive fingertip forces/torques to prevent overcompensation
    ft_force_norm = torch.norm(vec_sensor_tensor.view(vec_sensor_tensor.shape[0], -1), dim=-1)
    ft_penalty = -0.0001 * ft_force_norm

    # NEW: Penalize excessive joint torques
    torque_penalty = -0.00001 * torch.sum(dof_force_tensor ** 2, dim=-1)

    # Combine rewards
    total_reward = (
        rot_reward +
        0.5 * angvel_reward +
        action_reward +
        0.5 * drift_penalty +
        prox_reward +
        max_prox_penalty +
        time_bonus +
        acc_penalty +
        neutral_penalty +
        ft_penalty +
        torque_penalty
    )

    reward_components = {
        "rot_reward": rot_reward,
        "angvel_reward": angvel_reward,
        "action_penalty": action_reward,
        "drift_penalty": drift_penalty,
        "prox_reward": prox_reward,
        "max_prox_penalty": max_prox_penalty,
        "time_bonus": time_bonus,
        "acc_penalty": acc_penalty,
        "neutral_penalty": neutral_penalty,
        "ft_penalty": ft_penalty,
        "torque_penalty": torque_penalty
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
    y = w1 * x2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([x, y, z, w], dim=-1)

@torch.jit.script
def quat_conjugate(a: torch.Tensor) -> torch.Tensor:
    return torch.cat([-a[..., :3], a[..., 3:4]], dim=-1)