def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    shadow_hand_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error using quaternion distance
    # Rotation error: angle between object and goal orientation
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    rot_error = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:], dim=-1), min=0.0, max=1.0))
    
    # Reward for aligning orientation (higher reward for smaller rotation error)
    rot_reward_temp = 1.0
    rot_reward = torch.exp(-rot_error / rot_reward_temp)
    
    # Angular velocity regularization: penalize excessive spinning beyond what's needed
    angvel_norm = torch.norm(object_angvel, dim=-1)
    angvel_temp = 2.0
    angvel_reward = torch.exp(-angvel_norm / angvel_temp)
    
    # Action regularization: penalize large actions for energy efficiency
    action_norm = torch.norm(actions, dim=-1)
    action_temp = 0.1
    action_penalty = torch.exp(-action_norm / action_temp)
    
    # Joint velocity regularization: prevent jittery movements
    dof_vel_norm = torch.norm(shadow_hand_dof_vel, dim=-1)
    dof_vel_temp = 1.0
    dof_vel_penalty = torch.exp(-dof_vel_norm / dof_vel_temp)
    
    # Combine rewards with weights
    total_reward = (
        2.0 * rot_reward +
        0.5 * angvel_reward +
        0.3 * action_penalty +
        0.2 * dof_vel_penalty
    )
    
    reward_components = {
        "rot_reward": rot_reward,
        "angvel_reward": angvel_reward,
        "action_penalty": action_penalty,
        "dof_vel_penalty": dof_vel_penalty
    }
    
    return total_reward, reward_components


# Helper functions required for TorchScript compatibility
@torch.jit.script
def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[-1] == 4
    assert b.shape[-1] == 4

    x1, y1, z1, w1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    x2, y2, z2, w2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return torch.stack([x, y, z, w], dim=-1)

@torch.jit.script
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([-q[..., :3], q[..., 3:]], dim=-1)