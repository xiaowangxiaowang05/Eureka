def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    shadow_hand_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error using quaternion distance
    rel_rot = quat_mul(object_rot, quat_conjugate(goal_rot))
    # Use norm of vector part to get sin(theta/2), then compute angle
    rot_error = 2.0 * torch.asin(torch.clamp(torch.norm(rel_rot[:, :3], p=2, dim=-1), max=1.0))
    
    # Dense orientation reward: higher when closer to target orientation
    orientation_temp = 2.0
    orientation_reward = torch.exp(-orientation_temp * rot_error)
    
    # Encourage sustained rotation (non-zero angular velocity) to promote spinning
    angvel_norm = torch.norm(object_angvel, dim=-1)
    angvel_target = 1.5  # moderate spin speed
    angvel_error = torch.abs(angvel_norm - angvel_target)
    angvel_temp = 1.0
    angvel_reward = torch.exp(-angvel_temp * angvel_error)
    
    # Light action regularization to avoid jitter, but not so strong it prevents motion
    action_penalty = torch.sum(actions**2, dim=-1)
    action_reg_weight = 0.001
    action_reward = -action_reg_weight * action_penalty

    # Total reward combines orientation alignment, spin encouragement, and mild action cost
    total_reward = (
        orientation_reward +
        0.8 * angvel_reward +
        action_reward
    )

    reward_components = {
        "orientation_reward": orientation_reward,
        "angvel_reward": angvel_reward,
        "action_reward": action_reward
    }

    return total_reward, reward_components


# Helper functions for TorchScript compatibility
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