def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute rotational difference as angle between object and goal orientations
    # quat_diff = object_rot * conjugate(goal_rot)
    goal_rot_conj = torch.cat([goal_rot[:, 0:1], -goal_rot[:, 1:]], dim=1)
    quat_diff = quat_mul(object_rot, goal_rot_conj)
    
    # Extract vector part and compute angular error: theta = 2 * arcsin(||v||)
    # Clamp for numerical stability
    sin_half_theta = torch.clamp(torch.norm(quat_diff[:, 1:], p=2, dim=-1), max=1.0)
    rot_error = 2.0 * torch.asin(sin_half_theta)  # in [0, pi]
    
    # Use exponential reward that peaks sharply at 0 error
    temp_rot = 5.0  # temperature parameter for orientation reward
    rot_reward = torch.exp(-temp_rot * rot_error)
    
    total_reward = rot_reward
    
    reward_dict = {
        "rot_error": rot_error,
        "rot_reward": rot_reward
    }
    
    return total_reward, reward_dict


# Helper function for quaternion multiplication (TorchScript-compatible)
@torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([x, y, z, w], dim=-1)