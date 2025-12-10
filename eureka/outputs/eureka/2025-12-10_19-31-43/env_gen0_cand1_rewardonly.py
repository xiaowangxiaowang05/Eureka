def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, object_angvel: torch.Tensor, progress_buf: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute rotational error using quaternion distance
    # The rotation error quaternion is: object_rot * conjugate(goal_rot)
    # The scalar part (w component) of this quaternion relates to the angle between rotations
    rot_error_quat = quat_mul(object_rot, quat_conjugate(goal_rot))
    
    # Extract the w component (cos(theta/2) where theta is the rotation angle)
    rot_error_w = rot_error_quat[:, 3]
    
    # Clamp to avoid numerical issues
    rot_error_w = torch.clamp(rot_error_w, -1.0, 1.0)
    
    # Compute the rotation angle error (theta = 2 * acos(|w|))
    # We use absolute value since we care about magnitude of error
    rot_angle_error = 2.0 * torch.acos(torch.abs(rot_error_w))
    
    # Alternative approach: use 1 - |w| as a measure of rotational alignment
    # This is smoother near zero error
    rot_alignment = torch.abs(rot_error_w)
    
    # Temperature parameter for exponential reward shaping
    rot_temp = 1.0
    
    # Reward for rotational alignment (higher when closer to target)
    rot_reward = torch.exp(-rot_temp * rot_angle_error)
    
    # Additional reward component for appropriate angular velocity
    # We want some angular velocity to enable spinning, but not too much
    angvel_norm = torch.norm(object_angvel, dim=1)
    angvel_temp = 0.1
    # Encourage moderate angular velocity (not too slow, not too fast)
    optimal_angvel = 2.0  # target angular velocity magnitude
    angvel_error = torch.abs(angvel_norm - optimal_angvel)
    angvel_reward = torch.exp(-angvel_temp * angvel_error)
    
    # Time penalty to encourage faster completion
    time_penalty = -0.01 * progress_buf.float()
    
    # Combine rewards
    total_reward = rot_reward + 0.1 * angvel_reward + time_penalty
    
    reward_components = {
        "rot_reward": rot_reward,
        "angvel_reward": angvel_reward,
        "time_penalty": time_penalty
    }
    
    return total_reward, reward_components

# Helper functions for quaternion operations
@torch.jit.script
def quat_conjugate(q):
    return torch.cat([-q[:, :3], q[:, 3:4]], dim=1)

@torch.jit.script
def quat_mul(q1, q2):
    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return torch.stack([x, y, z, w], dim=1)