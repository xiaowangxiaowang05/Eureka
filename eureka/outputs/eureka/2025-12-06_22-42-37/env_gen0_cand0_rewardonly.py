@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, object_angvel: torch.Tensor, progress_buf: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error using quaternion dot product
    # Account for quaternion double cover by taking absolute value
    rot_error = 1.0 - torch.abs(torch.sum(object_rot * goal_rot, dim=-1))
    
    # Temperature parameter for orientation reward
    rot_temp = 1.0
    rot_reward = torch.exp(-rot_error / rot_temp)
    
    # Penalize high angular velocity to encourage stable final orientation
    angvel_penalty = torch.sum(object_angvel * object_angvel, dim=-1)
    angvel_temp = 0.1
    angvel_reward = torch.exp(-angvel_penalty / angvel_temp)
    
    # Combine rewards
    total_reward = rot_reward * angvel_reward
    
    reward_components = {
        "rot_reward": rot_reward,
        "angvel_reward": angvel_reward,
        "rot_error": rot_error,
        "angvel_penalty": angvel_penalty
    }
    
    return total_reward, reward_components
