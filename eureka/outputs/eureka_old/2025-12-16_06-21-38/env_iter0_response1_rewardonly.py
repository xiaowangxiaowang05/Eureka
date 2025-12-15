@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the rotation error between object and goal orientation
    # quat_mul(a, b) gives a * b; to get relative rotation from object to goal: q_error = goal * inv(object)
    # But here we already have quat_mul(object_rot, quat_conjugate(goal_rot)) in obs, 
    # so we can compute angle error from that, but since it's not passed in, we recompute.
    
    # Compute relative quaternion: object_rot^{-1} * goal_rot
    quat_diff = quat_mul(quat_conjugate(object_rot), goal_rot)
    
    # Extract angle of rotation error (magnitude of angular difference)
    # The w component of the quaternion is cos(theta/2), so theta = 2 * acos(|w|)
    # Clamp for numerical stability
    w = quat_diff[:, 3]
    w = torch.clamp(w, -1.0, 1.0)
    rot_error = 2.0 * torch.acos(torch.abs(w))  # in [0, pi]

    # Use exponential reward based on rotation error
    rot_reward_temp = 1.0
    rot_reward = torch.exp(-rot_reward_temp * rot_error)

    reward = rot_reward

    return reward, {
        "rot_reward": rot_reward,
        "rot_error": rot_error
    }