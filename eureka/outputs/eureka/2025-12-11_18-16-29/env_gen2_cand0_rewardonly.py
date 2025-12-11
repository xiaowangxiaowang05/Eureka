def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_pos: torch.Tensor,
    fingertip_pos: torch.Tensor,
    actions: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute rotational distance using angle between quaternions
    rel_quat = quat_mul(object_rot, quat_conjugate(goal_rot))
    rot_angle = 2.0 * torch.asin(torch.clamp(torch.norm(rel_quat[:, :3], p=2, dim=-1), max=1.0))  # [0, pi]
    
    # Positional stability: keep object near initial height and x-y center
    pos_error_xy = torch.norm(object_pos[:, :2], dim=1)
    pos_error_z = torch.abs(object_pos[:, 2] - 0.5)  # nominal z ~0.5 from env setup
    pos_error = pos_error_xy + pos_error_z

    # Encourage fingertips to stay close to the object (maintain grasp)
    # fingertip_pos shape: (num_envs, num_fingertips, 3)
    dist_fingertip_to_object = torch.norm(fingertip_pos - object_pos.unsqueeze(1), dim=2)  # (num_envs, num_fingertips)
    contact_reward = torch.mean(dist_fingertip_to_object, dim=1)

    # Action regularization
    action_penalty = torch.sum(actions ** 2, dim=1)

    # Temperature parameters for scaling
    rot_temp: float = 0.25
    pos_temp: float = 0.2
    contact_temp: float = 0.05
    action_temp: float = 0.01

    # Rewards / penalties (higher is better)
    rot_reward = torch.exp(-rot_angle / rot_temp)
    pos_penalty = torch.exp(-pos_error / pos_temp)
    contact_bonus = torch.exp(-contact_reward / contact_temp)
    action_reg = torch.exp(-action_penalty / action_temp)

    # Total reward combines alignment, stability, contact, and smoothness
    reward = rot_reward * pos_penalty * contact_bonus * action_reg

    reward_components = {
        "rot_reward": rot_reward,
        "pos_penalty": pos_penalty,
        "contact_bonus": contact_bonus,
        "action_reg": action_reg
    }

    return reward, reward_components

# Reuse helper functions from parent code (required for TorchScript compatibility)
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