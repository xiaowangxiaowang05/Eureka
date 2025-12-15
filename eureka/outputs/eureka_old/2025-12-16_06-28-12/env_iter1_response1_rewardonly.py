@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the rotation error between object and goal orientation
    # quat_mul(q, quat_conjugate(p)) gives the relative rotation from p to q
    rot_error = quat_mul(object_rot, quat_conjugate(goal_rot))
    
    # The angle of rotation error can be extracted from the w component:
    # For a unit quaternion [x, y, z, w], the angle is 2 * acos(|w|)
    # To get a smooth reward, we use 1 - |w| or similar; however, better to use the full angle
    # But for differentiability and boundedness, we use 1 - sqrt(x^2 + y^2 + z^2) = |w| (since unit quat)
    # Actually, distance on SO(3) can be approximated by 1 - |w|
    rot_dist = 1.0 - torch.abs(rot_error[:, 3])  # w component

    # Temperature parameter for exponential transformation
    rot_temp = 1.0
    rot_reward = torch.exp(-rot_dist / rot_temp)

    reward = rot_reward
    reward_components = {
        "rot_reward": rot_reward
    }

    return reward, reward_components