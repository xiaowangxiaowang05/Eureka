@torch.jit.script
def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_linvel: torch.Tensor,
    object_angvel: torch.Tensor,
    progress_buf: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error using quaternion distance
    # The dot product of two unit quaternions gives cos(theta/2) where theta is the rotation angle between them
    quat_diff = torch.abs(torch.sum(object_rot * goal_rot, dim=-1))
    # Clamp to valid range to avoid numerical issues
    quat_diff = torch.clamp(quat_diff, 0.0, 1.0)
    # Convert to angle error (smaller angle = better)
    orientation_error = 1.0 - quat_diff
    
    # Temperature parameter for orientation reward transformation
    orientation_temp = 1.0
    orientation_reward = torch.exp(-orientation_error / orientation_temp)
    
    # Penalize linear velocity to keep object stable
    linvel_penalty = torch.sum(object_linvel ** 2, dim=-1)
    linvel_temp = 0.1
    linvel_reward = torch.exp(-linvel_penalty / linvel_temp)
    
    # Total reward combines orientation alignment and stability
    total_reward = orientation_reward * linvel_reward
    
    reward_components = {
        "orientation_reward": orientation_reward,
        "linvel_reward": linvel_reward,
        "orientation_error": orientation_error,
        "linvel_penalty": linvel_penalty
    }
    
    return total_reward, reward_components
