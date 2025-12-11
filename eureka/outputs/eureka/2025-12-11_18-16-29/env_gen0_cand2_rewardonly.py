def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the rotation error between object and goal orientation
    # quat_mul(q1, quat_conjugate(q2)) gives relative rotation from q2 to q1
    rot_error_quat = quat_mul(object_rot, quat_conjugate(goal_rot))
    
    # Extract w component (real part); closer to 1 means smaller angle
    w = rot_error_quat[:, 3]
    
    # Clamp to avoid numerical issues
    w = torch.clamp(w, -1.0, 1.0)
    
    # Compute angular error in radians: theta = 2 * acos(|w|)
    # But since we want reward, higher when aligned => use |w|
    rot_error = 1.0 - torch.abs(w)

    # Temperature parameter for exponential shaping
    rot_temp: float = 1.0
    
    # Exponential reward shaping to encourage precise alignment
    rot_reward = torch.exp(-rot_error / rot_temp)

    reward = rot_reward
    reward_components = {
        "rot_reward": rot_reward
    }

    return reward, reward_components


# Helper functions needed for TorchScript compatibility
@torch.jit.script
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    shape = q.shape
    q = q.view(-1, 4)
    q_conj = torch.cat([q[:, :3] * -1, q[:, 3:]], dim=-1)
    return q_conj.view(shape)

@torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    shape = q1.shape
    q1 = q1.view(-1, 4)
    q2 = q2.view(-1, 4)
    
    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    
    quat = torch.stack([x, y, z, w], dim=-1)
    return quat.view(shape)