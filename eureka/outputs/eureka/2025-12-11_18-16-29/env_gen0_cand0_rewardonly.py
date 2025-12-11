def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the rotation error between object and goal orientations
    # Use quaternion difference: q_error = q_obj * q_goal^{-1}
    # The norm of the vector part of the error quaternion relates to angular error
    goal_rot_conj = torch.cat([goal_rot[:, 0:1], -goal_rot[:, 1:]], dim=1)
    rot_error_quat = quat_mul(object_rot, goal_rot_conj)
    
    # Extract the scalar (w) component; higher w means smaller rotation error
    w = rot_error_quat[:, 0]
    
    # Clamp to [-1, 1] to avoid numerical issues
    w = torch.clamp(w, -1.0, 1.0)
    
    # Angular error in radians: theta = 2 * acos(|w|)
    # But since reward should be higher when error is lower, we use cos(theta/2) = |w|
    # So we can directly use |w| as a proximity measure (max at 1 when aligned)
    rot_reward = torch.abs(w)
    
    # Optional: temperature-scaled exponential reward for sharper signal
    temp_rot = 1.0  # temperature parameter for orientation reward
    rot_reward_exp = torch.exp(temp_rot * (rot_reward - 1.0))  # peaks at 1 when rot_reward=1
    
    total_reward = rot_reward_exp
    
    reward_dict = {
        "rot_reward": rot_reward,
        "rot_reward_exp": rot_reward_exp
    }
    
    return total_reward, reward_dict


# Helper functions needed for TorchScript compatibility
@torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    assert q1.shape[-1] == 4 and q2.shape[-1] == 4
    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([x, y, z, w], dim=-1)