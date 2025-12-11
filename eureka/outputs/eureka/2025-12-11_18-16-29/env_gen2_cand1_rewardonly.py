def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    fingertip_pos: torch.Tensor,
    object_pos: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute rotation error as geodesic distance on SO(3)
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))  # [0, pi]

    # Encourage spinning around the correct axis once aligned
    # Compute target rotation axis from goal orientation
    z_axis = torch.zeros_like(object_pos)
    z_axis[:, 2] = 1.0
    goal_z_axis = quat_apply(goal_rot, z_axis)  # desired spin axis in world frame
    angvel_proj = torch.sum(object_angvel * goal_z_axis, dim=1)  # projection onto desired axis
    # Encourage high angular speed in the correct direction
    spin_reward = torch.clamp(angvel_proj, min=0.0)  # only reward positive spin

    # Penalize object drifting away from hand workspace (xy-plane)
    pos_error = torch.norm(object_pos[:, :2], dim=1)
    
    # Encourage fingertips to stay close to object (implicit grasp)
    fingertip_object_dist = torch.norm(fingertip_pos - object_pos.unsqueeze(1), dim=2).mean(dim=1)

    # Temperature parameters for exponential scaling
    rot_temp: float = 0.5
    spin_temp: float = 2.0
    pos_temp: float = 1.0
    contact_temp: float = 0.5

    # Use exponential rewards for sharp signal near success
    rot_reward = torch.exp(-rot_dist / rot_temp)
    spin_reward_scaled = torch.exp(spin_reward / spin_temp)
    pos_penalty = torch.exp(-pos_error / pos_temp)
    contact_reward = torch.exp(-fingertip_object_dist / contact_temp)

    # Total reward: strong emphasis on rotation alignment and correct spinning
    reward = rot_reward * spin_reward_scaled * pos_penalty * contact_reward

    reward_components = {
        "rot_reward": rot_reward,
        "spin_reward": spin_reward_scaled,
        "pos_penalty": pos_penalty,
        "contact_reward": contact_reward
    }

    return reward, reward_components


# TorchScript-compatible helper functions
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

@torch.jit.script
def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    shape = q.shape
    q = q.reshape(-1, 4)
    v = v.reshape(-1, 3)
    xyz = q[:, :3]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    result = v + q[:, 3:4] * t + torch.cross(xyz, t, dim=-1)
    return result.reshape(shape[:-1] + (3,))