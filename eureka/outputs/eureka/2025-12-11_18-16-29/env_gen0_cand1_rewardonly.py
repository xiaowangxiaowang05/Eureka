def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    fingertip_pos: torch.Tensor,
    object_pos: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error using quaternion distance
    # quat_mul(q1, quat_conjugate(q2)) gives the relative rotation from q2 to q1
    rel_quat = quat_mul(object_rot, quat_conjugate(goal_rot))
    # The w component of the relative quaternion relates to the angle difference
    rot_error = 1.0 - torch.abs(rel_quat[:, 0])  # 1 - |w|, ranges [0, 1]
    
    # Angular velocity regularization (penalize excessive spinning once aligned)
    angvel_norm = torch.norm(object_angvel, dim=1)
    
    # Object should stay near center (penalize drifting)
    pos_error = torch.norm(object_pos[:, :2], dim=1)  # ignore z for tabletop tasks
    
    # Encourage fingertips to stay near object (implicit contact)
    fingertip_object_dist = torch.norm(fingertip_pos - object_pos.unsqueeze(1), dim=2).mean(dim=1)
    
    # Temperature parameters for exponential scaling
    rot_temp: float = 1.0
    angvel_temp: float = 0.1
    pos_temp: float = 1.0
    contact_temp: float = 1.0

    rot_reward = torch.exp(-rot_error / rot_temp)
    angvel_penalty = torch.exp(-angvel_norm / angvel_temp)
    pos_penalty = torch.exp(-pos_error / pos_temp)
    contact_reward = torch.exp(-fingertip_object_dist / contact_temp)

    # Total reward is a combination of alignment, stability, position, and contact
    reward = rot_reward * angvel_penalty * pos_penalty * contact_reward

    reward_components = {
        "rot_reward": rot_reward,
        "angvel_penalty": angvel_penalty,
        "pos_penalty": pos_penalty,
        "contact_reward": contact_reward
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