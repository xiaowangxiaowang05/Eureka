@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the rotation error between object and goal orientation
    # quat_mul(q, quat_conjugate(p)) gives the relative rotation from p to q
    rot_error = quat_mul(object_rot, quat_conjugate(goal_rot))
    
    # The angle of rotation error can be derived from the w component (real part)
    # since for a unit quaternion, angle = 2 * acos(|w|), but we can use 1 - |w| as a proxy
    # We take absolute value to handle sign ambiguity in quaternions (q and -q represent same rotation)
    rot_dist = 1.0 - torch.abs(rot_error[:, 3])  # rot_error[:, 3] is the 'w' component

    # Temperature parameter for exponential transformation
    rot_temp = 1.0
    rot_reward = torch.exp(-rot_dist / rot_temp)

    reward = rot_reward
    reward_components = {
        "rot_reward": rot_reward,
        "rot_dist": rot_dist
    }

    return reward, reward_components