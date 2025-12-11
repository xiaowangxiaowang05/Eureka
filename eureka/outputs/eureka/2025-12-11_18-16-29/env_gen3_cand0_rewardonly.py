def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    fingertip_pos: torch.Tensor,
    object_pos: torch.Tensor,
    actions: torch.Tensor,
    success_tolerance: float,
    reach_goal_bonus: float,
    action_penalty_scale: float
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute rotational distance using angle of relative quaternion
    rel_quat = quat_mul(object_rot, quat_conjugate(goal_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(rel_quat[:, :3], p=2, dim=-1), max=1.0))

    # Orientation reward: higher when closer to target
    rot_temp: float = 0.5
    rot_reward = torch.exp(-rot_dist / rot_temp)

    # Bonus for successful alignment
    success = (rot_dist <= success_tolerance).float()
    success_bonus = success * reach_goal_bonus

    # Penalize large actions
    action_penalty = action_penalty_scale * torch.sum(actions ** 2, dim=-1)

    # Encourage contact by keeping fingertips near object
    fingertip_object_dist = torch.norm(fingertip_pos - object_pos.unsqueeze(1), dim=2).mean(dim=1)
    contact_temp: float = 0.1
    contact_reward = torch.exp(-fingertip_object_dist / contact_temp)

    # Total reward
    reward = rot_reward + success_bonus + contact_reward - action_penalty

    reward_components = {
        "rot_reward": rot_reward,
        "success_bonus": success_bonus,
        "contact_reward": contact_reward,
        "action_penalty": -action_penalty
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