@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the rotation error between object and goal orientation
    # quat_mul(q, quat_conjugate(p)) gives the relative rotation from p to q
    rot_error = quat_mul(object_rot, quat_conjugate(goal_rot))
    
    # Extract the angle of the rotation error (using the scalar part of the quaternion)
    # The scalar (w) component relates to the angle via w = cos(theta/2)
    # So the angular error magnitude is 2 * acos(|w|), but we can use 1 - |w| as a proxy for small angles
    rot_error_angle = 1.0 - torch.abs(rot_error[:, 3])  # w component is at index 3

    # Temperature parameter for exponential reward shaping
    rot_temp = 1.0

    # Exponentially shaped orientation reward: higher when orientation is closer
    orientation_reward = torch.exp(-rot_temp * rot_error_angle)

    # Total reward is just the orientation alignment reward
    total_reward = orientation_reward

    return total_reward, {
        "orientation_reward": orientation_reward,
        "rot_error_angle": rot_error_angle
    }