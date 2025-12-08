def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, object_linvel: torch.Tensor, object_angvel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error using quaternion difference
    # quat_mul(object_rot, quat_conjugate(goal_rot)) gives the relative rotation
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    
    # The scalar part (w component) of the quaternion difference indicates alignment
    # When orientations match perfectly, quat_diff.w = 1
    # When orientations are opposite, quat_diff.w = 0
    orientation_error = 1.0 - torch.abs(quat_diff[:, 0])  # Use absolute value to handle quaternion double cover
    
    # Penalize linear velocity to keep object stable
    linvel_penalty = torch.sum(object_linvel**2, dim=1)
    
    # Temperature parameters for reward shaping
    orientation_temp = 10.0
    linvel_temp = 0.1
    
    # Compute individual reward components
    orientation_reward = torch.exp(-orientation_temp * orientation_error)
    linvel_reward = torch.exp(-linvel_temp * linvel_penalty)
    
    # Total reward
    total_reward = orientation_reward * linvel_reward
    
    reward_components = {
        "orientation_reward": orientation_reward,
        "linvel_reward": linvel_reward
    }
    
    return total_reward, reward_components

@torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions"""
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return torch.stack([w, x, y, z], dim=1)

@torch.jit.script
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Compute quaternion conjugate"""
    return torch.cat([q[:, 0:1], -q[:, 1:]], dim=1)
