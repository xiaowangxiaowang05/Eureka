@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the rotation error between object and goal orientation
    # quat_mul(a, b) gives a * b; to get relative rotation from goal to object: object_rot * conj(goal_rot)
    rel_quat = quat_mul(object_rot, quat_conjugate(goal_rot))
    
    # The angle of the relative rotation (w component of quaternion relates to angle)
    # Rotation error: 1 - |w| encourages alignment (max |w| = 1 when aligned)
    rot_error = 1.0 - torch.abs(rel_quat[:, 3])  # w is the last component

    # Temperature for exponential reward shaping
    rot_temp = 1.0

    # Use exponential reward to encourage precise orientation matching
    rot_reward = torch.exp(-rot_error / rot_temp)

    reward = rot_reward

    return reward, {
        "rot_reward": rot_reward,
        "rot_error": rot_error
    }