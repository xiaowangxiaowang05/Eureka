@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the rotation error between object and goal orientation
    # quat_mul(q1, quat_conjugate(q2)) gives the relative rotation from q2 to q1
    rot_error_quat = quat_mul(object_rot, quat_conjugate(goal_rot))
    
    # The w component of the error quaternion relates to the angle of rotation
    # Error is minimized when rot_error_quat[..., 0] (w) approaches 1
    rot_error_angle = 2.0 * torch.acos(torch.clamp(rot_error_quat[:, 0], -1.0, 1.0))
    
    # Use exponential reward based on angular error
    rot_reward_temp = 1.0  # temperature parameter for orientation reward
    rot_reward = torch.exp(-rot_error_angle / rot_reward_temp)
    
    # Total reward is just the orientation alignment reward
    reward = rot_reward
    
    reward_components = {
        "rot_reward": rot_reward,
        "rot_error_angle": rot_error_angle
    }
    
    return reward, reward_components