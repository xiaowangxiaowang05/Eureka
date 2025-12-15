@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the rotation error between object and goal orientation
    # quat_mul(q1, quat_conjugate(q2)) gives the relative rotation from q2 to q1
    rot_error_quat = quat_mul(object_rot, quat_conjugate(goal_rot))
    
    # The w component of the error quaternion relates to the angle of rotation
    # Rotation error in angle: theta = 2 * acos(|w|), but we can use 1 - |w| as a smooth proxy
    rot_error = 1.0 - torch.abs(rot_error_quat[:, 3])  # w component is at index 3

    # Temperature parameter for exponential shaping
    rot_temp = 1.0
    
    # Exponentially shaped reward to encourage precise alignment
    rot_reward = torch.exp(-rot_temp * rot_error)

    reward = rot_reward

    return reward, {
        "rot_reward": rot_reward,
        "rot_error": rot_error
    }