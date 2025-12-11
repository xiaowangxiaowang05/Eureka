def compute_reward(
    object_pos: torch.Tensor,
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    fingertip_pos: torch.Tensor,
    actions: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute rotation error as the angle of the relative rotation (geodesic distance on SO(3))
    rel_quat = quat_mul(object_rot, quat_conjugate(goal_rot))
    # Ensure w is in [-1, 1] before taking arccos
    w = torch.clamp(rel_quat[:, 3], -1.0, 1.0)
    rot_angle = 2.0 * torch.acos(torch.abs(w))  # [0, pi]
    
    # Position stability: keep object near origin in XY plane
    pos_xy_dist = torch.norm(object_pos[:, :2], dim=1)
    
    # Contact: fingertips should stay close to object
    # fingertip_pos shape: [num_envs, num_fingertips, 3]
    fingertip_object_dist = torch.norm(fingertip_pos - object_pos.unsqueeze(1), dim=2).mean(dim=1)
    
    # Action regularization
    action_penalty = torch.sum(actions ** 2, dim=1)
    
    # Temperature parameters for exponential scaling
    rot_temp: float = 0.5
    pos_temp: float = 0.1
    contact_temp: float = 0.05
    action_temp: float = 0.01

    # Use exponential rewards to create strong gradient near target
    rot_reward = torch.exp(-rot_angle / rot_temp)
    pos_penalty = torch.exp(-pos_xy_dist / pos_temp)
    contact_reward = torch.exp(-fingertip_object_dist / contact_temp)
    action_reg = torch.exp(-action_penalty / action_temp)

    # Total reward combines orientation alignment, position stability, contact, and smooth actions
    reward = rot_reward * pos_penalty * contact_reward * action_reg

    reward_components = {
        "rot_reward": rot_reward,
        "pos_penalty": pos_penalty,
        "contact_reward": contact_reward,
        "action_reg": action_reg
    }

    return reward, reward_components


# Helper functions required by TorchScript
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