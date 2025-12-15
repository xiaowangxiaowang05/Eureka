@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the rotation error between object and goal orientation
    # quat_mul(q, quat_conjugate(p)) gives the relative rotation from p to q
    rot_error = quat_mul(object_rot, quat_conjugate(goal_rot))
    
    # The angle of rotation error can be derived from the w component:
    # For a unit quaternion [x, y, z, w], the rotation angle is 2 * acos(|w|)
    # To avoid instability near |w| = 1, we clamp the value
    w = rot_error[:, 3]  # w component of the relative rotation
    w_clamped = torch.clamp(w, -1.0, 1.0)
    rot_angle_error = 2.0 * torch.acos(torch.abs(w_clamped))

    # Use exponential reward to encourage small orientation errors
    rot_reward_temp = 0.5  # temperature parameter for orientation reward
    rot_reward = torch.exp(-rot_angle_error / rot_reward_temp)

    reward = rot_reward

    return reward, {
        "rot_reward": rot_reward,
        "rot_angle_error": rot_angle_error
    }