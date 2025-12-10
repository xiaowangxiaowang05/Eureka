@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, object_angvel: torch.Tensor, object_linvel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error using quaternion dot product
    # Quaternion distance: 1 - |dot(q1, q2)|, where perfect alignment = 0
    quat_diff = torch.abs(torch.sum(object_rot * goal_rot, dim=-1))
    orientation_error = 1.0 - quat_diff
    
    # Penalize linear velocity to keep object stable in position
    linvel_penalty = torch.norm(object_linvel, dim=-1)
    
    # Optionally, we could reward appropriate angular velocity for spinning
    # But since we want to reach a specific orientation, we might want to penalize excessive angular velocity
    angvel_penalty = torch.norm(object_angvel, dim=-1)
    
    # Temperature parameters for exponential transformations
    orientation_temp = 10.0
    linvel_temp = 0.1
    angvel_temp = 0.1
    
    # Transform rewards using exponential to normalize ranges
    orientation_reward = torch.exp(-orientation_temp * orientation_error)
    linvel_reward = torch.exp(-linvel_temp * linvel_penalty)
    angvel_reward = torch.exp(-angvel_temp * angvel_penalty)
    
    # Combine rewards
    total_reward = orientation_reward * linvel_reward * angvel_reward
    
    reward_components = {
        "orientation_reward": orientation_reward,
        "linvel_reward": linvel_reward,
        "angvel_reward": angvel_reward
    }
    
    return total_reward, reward_components