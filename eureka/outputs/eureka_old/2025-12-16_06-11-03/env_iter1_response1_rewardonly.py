@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the rotation error between object and goal orientation
    # quat_mul(a, b) gives a * b; to get relative rotation from object to goal: q_error = goal * inv(object)
    # Inverse of a unit quaternion is its conjugate
    quat_error = quat_mul(goal_rot, quat_conjugate(object_rot))
    
    # The scalar (w) component of the error quaternion relates to the angle of rotation needed
    # The closer |w| is to 1, the smaller the angular error
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_error[:, 1:], dim=1), min=0.0, max=1.0))  # angular distance in radians
    
    # Temperature parameter for exponential shaping
    rot_temp = 1.0
    
    # Exponential reward based on rotational alignment (higher when aligned)
    rot_reward = torch.exp(-rot_dist / rot_temp)
    
    # Total reward
    reward = rot_reward
    
    return reward, {
        "rot_reward": rot_reward,
        "rot_dist": rot_dist
    }