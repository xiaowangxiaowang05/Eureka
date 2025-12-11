def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    fingertip_pos: torch.Tensor,
    object_pos: torch.Tensor,
    goal_pos: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error using quaternion distance
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    rot_error = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:], dim=1), min=0.0, max=1.0))
    
    # Reward for achieving target orientation (higher reward for smaller error)
    rot_reward_temp = 1.0
    rot_reward = torch.exp(-rot_error / rot_reward_temp)

    # Encourage spinning by rewarding angular velocity aligned with the rotation axis
    # For general spinning, we can reward magnitude of angular velocity
    angvel_norm = torch.norm(object_angvel, dim=1)
    angvel_reward_temp = 2.0
    angvel_reward = torch.exp(-torch.abs(angvel_norm - 5.0) / angvel_reward_temp)  # encourage moderate spin speed

    # Keep object near goal position (to avoid dropping or throwing)
    pos_error = torch.norm(object_pos - goal_pos, dim=1)
    pos_reward_temp = 0.1
    pos_reward = torch.exp(-pos_error / pos_reward_temp)

    # Encourage fingertip contact around the object (implicit via proximity)
    # Average distance from fingertips to object center
    fingertip_obj_dist = torch.norm(fingertip_pos - object_pos.unsqueeze(1), dim=2)  # [num_envs, num_fingertips]
    mean_fingertip_dist = torch.mean(fingertip_obj_dist, dim=1)
    contact_reward_temp = 0.1
    contact_reward = torch.exp(-mean_fingertip_dist / contact_reward_temp)

    # Total reward is a combination of all components
    total_reward = rot_reward + 0.5 * angvel_reward + 0.5 * pos_reward + 0.3 * contact_reward

    reward_components = {
        "rot_reward": rot_reward,
        "angvel_reward": angvel_reward,
        "pos_reward": pos_reward,
        "contact_reward": contact_reward
    }

    return total_reward, reward_components


# Helper functions needed for TorchScript compatibility
@torch.jit.script
def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape == b.shape
    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return torch.stack([x, y, z, w], dim=1).contiguous()

@torch.jit.script
def quat_conjugate(a: torch.Tensor) -> torch.Tensor:
    return torch.cat([-a[:, :3], a[:, 3:4]], dim=1).contiguous()