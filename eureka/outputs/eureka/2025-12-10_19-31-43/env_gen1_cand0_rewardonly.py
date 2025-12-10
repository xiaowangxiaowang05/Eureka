def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    progress_buf: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute rotation error via quaternion difference
    rot_error_quat = quat_mul(object_rot, quat_conjugate(goal_rot))
    rot_error_w = rot_error_quat[:, 3]
    rot_error_w = torch.clamp(rot_error_w, -1.0, 1.0)
    rot_angle_error = 2.0 * torch.acos(torch.abs(rot_error_w))

    # Rotational alignment reward with temperature scaling
    rot_temp = 2.0
    rot_reward = torch.exp(-rot_temp * rot_angle_error)

    # Angular velocity reward: encourage moderate spinning
    angvel_norm = torch.norm(object_angvel, dim=1)
    optimal_angvel = 2.0
    angvel_error = torch.abs(angvel_norm - optimal_angvel)
    angvel_temp = 0.5
    angvel_reward = torch.exp(-angvel_temp * angvel_error)

    # Action magnitude reward to prevent freezing: encourage non-zero actions
    action_magnitude = torch.norm(actions, dim=1)
    action_temp = 1.0
    action_reward = torch.exp(action_temp * (action_magnitude - 0.5))  # shifted to reward moderate activity

    # Combine rewards (no time penalty to avoid discouraging exploration)
    total_reward = rot_reward + 0.3 * angvel_reward + 0.2 * action_reward

    reward_components = {
        "rot_reward": rot_reward,
        "angvel_reward": angvel_reward,
        "action_reward": action_reward
    }

    return total_reward, reward_components

@torch.jit.script
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([-q[:, :3], q[:, 3:4]], dim=1)

@torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return torch.stack([x, y, z, w], dim=1)