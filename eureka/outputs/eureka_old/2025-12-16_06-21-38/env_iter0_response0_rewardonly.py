@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the rotation error between object and goal orientation
    # quat_mul(q1, quat_conjugate(q2)) gives the relative rotation from q2 to q1
    rot_error = quat_mul(object_rot, quat_conjugate(goal_rot))
    
    # Extract the angle of the rotation error (angle-axis representation)
    # The scalar part (w) of the quaternion is cos(theta/2)
    w = rot_error[:, 3]  # w component of the error quaternion
    # Clamp to avoid numerical issues
    w = torch.clamp(w, -1.0, 1.0)
    # Compute angle error: theta = 2 * acos(|w|), but we can use 1 - |w| as a smooth proxy
    angle_error = 1.0 - torch.abs(w)
    
    # Temperature parameter for exponential shaping
    rot_reward_temp = 1.0
    
    # Exponential reward based on alignment (higher when angle error is small)
    rot_reward = torch.exp(-rot_reward_temp * angle_error)
    
    # Total reward
    reward = rot_reward
    
    # Return reward and components
    reward_components = {
        "rot_reward": rot_reward,
        "angle_error": angle_error
    }
    
    return reward, reward_components