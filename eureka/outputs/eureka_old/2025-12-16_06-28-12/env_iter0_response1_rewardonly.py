@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the rotation error between object and goal orientation
    # quat_mul(a, b) gives a * b; to get relative rotation from goal to object: object_rot * conj(goal_rot)
    rel_quat = quat_mul(object_rot, quat_conjugate(goal_rot))
    
    # The angle of the relative rotation (w component corresponds to cos(theta/2))
    # We use 1 - |w| as a measure of misalignment (lower is better); alternatively, use norm of xyz part
    rot_error = torch.norm(rel_quat[:, 0:3], dim=1)  # sin(theta/2) magnitude; ranges [0, 1]
    
    # Temperature parameter for exponential reward shaping
    rot_reward_temp = 1.0
    
    # Exponential reward that peaks when rot_error is 0 (perfect alignment)
    rot_reward = torch.exp(-rot_reward_temp * rot_error)
    
    # Total reward is just the orientation alignment reward
    total_reward = rot_reward
    
    return total_reward, {
        "rot_reward": rot_reward,
        "rot_error": rot_error
    }