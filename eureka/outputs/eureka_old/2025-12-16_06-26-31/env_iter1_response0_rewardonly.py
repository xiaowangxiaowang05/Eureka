@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the rotation error between object and goal orientation
    # quat_mul(q1, quat_conjugate(q2)) gives the relative rotation from q2 to q1
    rot_error_quat = quat_mul(object_rot, quat_conjugate(goal_rot))
    
    # The scalar (w) component of the error quaternion relates to the angle difference
    # via: w = cos(theta/2), so |w| close to 1 means small rotation error
    rot_error_angle = 2.0 * torch.acos(torch.clamp(torch.abs(rot_error_quat[:, 3]), min=0.0, max=1.0))
    
    # Define temperature for exponential reward shaping
    rot_reward_temp: float = 1.0
    
    # Exponential reward based on angular error (smaller error → higher reward)
    rot_reward = torch.exp(-rot_reward_temp * rot_error_angle)
    
    # Total reward is just the orientation alignment reward
    total_reward = rot_reward

    return total_reward, {
        "rot_reward": rot_reward,
        "rot_error_angle": rot_error_angle
    }