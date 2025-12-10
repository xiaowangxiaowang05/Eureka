@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, object_angvel: torch.Tensor, progress_buf: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute quaternion error: q_error = object_rot * conjugate(goal_rot)
    # Quaternion conjugate flips the sign of imaginary components
    goal_rot_conj = torch.cat([goal_rot[:, 0:1], -goal_rot[:, 1:2], -goal_rot[:, 2:3], -goal_rot[:, 3:4]], dim=1)
    
    # Quaternion multiplication: q1 * q2
    w1, x1, y1, z1 = object_rot[:, 0], object_rot[:, 1], object_rot[:, 2], object_rot[:, 3]
    w2, x2, y2, z2 = goal_rot_conj[:, 0], goal_rot_conj[:, 1], goal_rot_conj[:, 2], goal_rot_conj[:, 3]
    
    q_error_w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    q_error_x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    q_error_y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    q_error_z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    q_error = torch.stack([q_error_w, q_error_x, q_error_y, q_error_z], dim=1)
    
    # The angle error can be computed from the scalar part of the quaternion error
    # angle = 2 * arccos(|w|), but we can use 1 - |w| as a proxy for small angles
    orientation_error = 1.0 - torch.abs(q_error[:, 0])
    
    # Temperature parameter for orientation reward
    orientation_temp = 10.0
    orientation_reward = torch.exp(-orientation_temp * orientation_error)
    
    # Penalty for excessive angular velocity (to encourage stable holding once oriented correctly)
    angvel_norm = torch.norm(object_angvel, dim=1)
    angvel_temp = 0.1
    angvel_penalty = torch.exp(-angvel_temp * angvel_norm)
    
    # Combine rewards
    reward = orientation_reward * angvel_penalty
    
    reward_components = {
        "orientation_reward": orientation_reward,
        "angvel_penalty": angvel_penalty,
        "orientation_error": orientation_error
    }
    
    return reward, reward_components