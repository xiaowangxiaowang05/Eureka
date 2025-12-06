@torch.jit.script
def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_linvel: torch.Tensor,
    object_angvel: torch.Tensor,
    progress_buf: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error using quaternion distance
    quat_diff = torch.abs(torch.sum(object_rot * goal_rot, dim=-1))
    quat_diff = torch.clamp(quat_diff, 0.0, 1.0)
    orientation_error = 1.0 - quat_diff
    
    # Temperature parameter for orientation reward transformation
    orientation_temp = 0.5
    orientation_reward = torch.exp(-orientation_error / orientation_temp)
    
    # Penalize linear velocity to keep object stable
    linvel_penalty = torch.sum(object_linvel ** 2, dim=-1)
    linvel_temp = 0.1
    linvel_reward = torch.exp(-linvel_penalty / linvel_temp)
    
    # Penalize angular velocity to reduce wobbling and overshooting near completion
    angvel_penalty = torch.sum(object_angvel ** 2, dim=-1)
    angvel_temp = 0.05
    angvel_reward = torch.exp(-angvel_penalty / angvel_temp)
    
    # Total reward combines orientation alignment, linear stability, and angular damping
    total_reward = orientation_reward * linvel_reward * angvel_reward
    
    reward_components = {
        "orientation_reward": orientation_reward,
        "linvel_reward": linvel_reward,
        "angvel_reward": angvel_reward,
        "orientation_error": orientation_error,
        "linvel_penalty": linvel_penalty,
        "angvel_penalty": angvel_penalty
    }
    
    return total_reward, reward_components
