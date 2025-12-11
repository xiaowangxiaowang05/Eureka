def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    object_pos: torch.Tensor,
    actions: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute rotation error as the geodesic distance on SO(3): angle of relative rotation
    rel_quat = quat_mul(object_rot, quat_conjugate(goal_rot))
    # Ensure scalar part (w) is in [-1, 1] for asin stability
    w = rel_quat[:, 3]
    w_clamped = torch.clamp(w, -1.0, 1.0)
    rot_angle = 2.0 * torch.acos(torch.abs(w_clamped))  # [0, pi], invariant to sign ambiguity
    
    # Penalize angular velocity only when close to target to allow spinning during reorientation
    angvel_norm = torch.norm(object_angvel, dim=1)
    angvel_penalty = torch.where(rot_angle < 0.5, angvel_norm, torch.zeros_like(angvel_norm))
    
    # Penalize object drifting in x-y plane (keep it centered)
    pos_drift = torch.norm(object_pos[:, :2], dim=1)
    
    # Action regularization
    action_penalty = torch.sum(actions ** 2, dim=1)

    # Temperature parameters for shaping rewards
    rot_temp: float = 0.25
    angvel_temp: float = 0.1
    pos_temp: float = 0.5
    action_temp: float = 0.01

    # Exponential rewards/penalties for smooth gradients
    rot_reward = torch.exp(-rot_angle / rot_temp)
    angvel_cost = torch.exp(-angvel_penalty / angvel_temp)
    pos_cost = torch.exp(-pos_drift / pos_temp)
    action_cost = torch.exp(-action_penalty / action_temp)

    # Combine components multiplicatively to require all conditions
    reward = rot_reward * angvel_cost * pos_cost * action_cost

    reward_components = {
        "rot_reward": rot_reward,
        "angvel_cost": angvel_cost,
        "pos_cost": pos_cost,
        "action_cost": action_cost
    }

    return reward, reward_components

@torch.jit.script
def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape == b.shape
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
    return torch.stack([x, y, z, w], dim=-1)

@torch.jit.script
def quat_conjugate(a: torch.Tensor) -> torch.Tensor:
    return torch.cat([-a[:, :3], a[:, 3:4]], dim=-1)