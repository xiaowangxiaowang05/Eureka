def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    shadow_hand_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error using quaternion distance
    # quat_mul(q1, quat_conjugate(q2)) gives relative rotation from q2 to q1
    rel_quat = quat_mul(object_rot, quat_conjugate(goal_rot))
    # Extract w component; closer to 1 means more aligned
    rot_dist = 1.0 - torch.abs(rel_quat[:, 3])  # [0, 1], lower is better
    
    # Angular velocity magnitude penalty (to avoid excessive spinning once aligned)
    angvel_mag = torch.norm(object_angvel, dim=-1)
    
    # Action regularization (smoothness and energy efficiency)
    action_norm = torch.norm(actions, dim=-1)
    
    # Joint velocity penalty for stability
    joint_vel_norm = torch.norm(shadow_hand_dof_vel, dim=-1)

    # Temperature parameters for reward shaping
    rot_temp: float = 2.0
    angvel_temp: float = 0.5
    action_temp: float = 0.05
    joint_vel_temp: float = 0.01

    # Shaped rewards
    rot_reward = torch.exp(-rot_temp * rot_dist)
    angvel_penalty = torch.exp(-angvel_temp * angvel_mag)
    action_penalty = torch.exp(-action_temp * action_norm)
    joint_vel_penalty = torch.exp(-joint_vel_temp * joint_vel_norm)

    # Total reward as product of components (all in [0,1])
    total_reward = rot_reward * angvel_penalty * action_penalty * joint_vel_penalty

    reward_components = {
        "rot_reward": rot_reward,
        "angvel_penalty": angvel_penalty,
        "action_penalty": action_penalty,
        "joint_vel_penalty": joint_vel_penalty
    }

    return total_reward, reward_components

# Helper functions required for TorchScript compatibility
@torch.jit.script
def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[-1] == 4
    assert b.shape[-1] == 4

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (y1 - z1) * (x2 - w2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1)
    return quat

@torch.jit.script
def quat_conjugate(a: torch.Tensor) -> torch.Tensor:
    return torch.cat([-a[:, :3], a[:, 3:4]], dim=-1)