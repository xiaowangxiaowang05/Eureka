@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the rotation error between object and goal orientation
    # quat_mul(q1, quat_conjugate(q2)) gives the relative rotation from q2 to q1
    rot_error = quat_mul(object_rot, quat_conjugate(goal_rot))
    
    # The w component of the relative rotation quaternion indicates alignment:
    # w = 1 means perfect alignment, w = 0 means 180-degree misalignment
    rot_dist = 1.0 - rot_error[:, 0] * rot_error[:, 0]  # 1 - w^2; ranges from 0 (aligned) to 1 (opposite)

    # Temperature parameter for exponential reward shaping
    rot_temp = 1.0
    
    # Exponentially shaped orientation reward: higher when rot_dist is small
    orientation_reward = torch.exp(-rot_temp * rot_dist)

    reward = orientation_reward

    return reward, {
        "orientation_reward": orientation_reward,
        "rot_dist": rot_dist
    }