@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the rotation error between object and goal orientation
    # quat_mul(q1, quat_conjugate(q2)) gives the relative rotation from q2 to q1
    rot_error_quat = quat_mul(object_rot, quat_conjugate(goal_rot))
    
    # Extract the angle of the rotation error (angle-axis representation)
    # The scalar part (w) of the quaternion relates to the rotation angle: w = cos(theta/2)
    rot_error_angle = 2.0 * torch.acos(torch.clamp(rot_error_quat[:, 3], -1.0, 1.0))  # shape: [num_envs]
    
    # Normalize the angle to [0, pi]
    rot_error_angle = torch.abs(rot_error_angle)
    
    # Use exponential reward based on angular error
    rot_reward_temp = 1.0  # temperature parameter for orientation reward
    rot_reward = torch.exp(-rot_error_angle / rot_reward_temp)
    
    # Total reward is just the orientation alignment reward
    total_reward = rot_reward
    
    reward_components = {
        "rot_reward": rot_reward,
        "rot_error_angle": rot_error_angle
    }
    
    return total_reward, reward_components