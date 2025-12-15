@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the rotation error between object and goal orientation
    # quat_mul(q1, quat_conjugate(q2)) gives the relative rotation from q2 to q1
    rot_error = quat_mul(object_rot, quat_conjugate(goal_rot))
    
    # Extract the angle of the rotation error (angle-axis representation)
    # The scalar part (w) of the quaternion relates to the rotation angle: w = cos(theta/2)
    # So theta = 2 * acos(|w|), but we can work directly with |w| for reward shaping
    rot_error_angle = 2.0 * torch.acos(torch.clamp(torch.abs(rot_error[:, 3]), min=0.0, max=1.0))
    
    # Reward is higher when rotation error is smaller
    # Use exponential reward with temperature for smoothness and better gradient signal
    rot_reward_temp = 1.0
    rot_reward = torch.exp(-rot_reward_temp * rot_error_angle)
    
    # Total reward
    reward = rot_reward
    
    # Return reward and components
    reward_components = {
        "rot_reward": rot_reward,
    }
    
    return reward, reward_components