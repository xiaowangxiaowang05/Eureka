@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the rotation error between object and goal orientation
    # quat_mul(a, b) gives a * b; to get relative rotation from goal to object: object_rot * conj(goal_rot)
    rel_quat = quat_mul(object_rot, quat_conjugate(goal_rot))
    
    # The angle of the relative rotation (from quaternion) is 2 * acos(|w|)
    # But we can use 1 - |w| as a smooth approximation of alignment (max when w=±1, i.e., aligned)
    rot_error = 1.0 - torch.abs(rel_quat[:, 3])  # w component

    # Temperature parameter for exponential shaping
    rot_temp = 1.0
    rot_reward = torch.exp(-rot_error / rot_temp)

    reward = rot_reward
    reward_components = {
        "rot_reward": rot_reward,
    }

    return reward, reward_components