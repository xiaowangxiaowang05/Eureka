@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the rotation error between object and goal orientation
    # quat_mul(q1, quat_conjugate(q2)) gives the relative rotation from q2 to q1
    rot_error_quat = quat_mul(object_rot, quat_conjugate(goal_rot))
    
    # The w component of the relative rotation quaternion relates to the angle difference
    # Specifically, angle = 2 * acos(|w|), so maximizing |w| minimizes the angle
    rot_error_cos = torch.abs(rot_error_quat[:, 3])  # absolute value of w component

    # Temperature parameter for exponential transformation
    rot_reward_temp = 1.0
    rot_reward = torch.exp(rot_error_cos / rot_reward_temp)

    reward = rot_reward

    return reward, {
        "rot_reward": rot_reward
    }