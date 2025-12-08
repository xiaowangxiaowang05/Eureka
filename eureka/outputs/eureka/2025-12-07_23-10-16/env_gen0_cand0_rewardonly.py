@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, object_angvel: torch.Tensor, object_linvel: torch.Tensor, progress_buf: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error using quaternion distance
    # The relative rotation quaternion: q_rel = object_rot * conjugate(goal_rot)
    # For unit quaternions, the angle error can be derived from the w component
    # Orientation error: smaller is better (0 when perfectly aligned)
    
    # Compute quaternion difference
    # Using the fact that for unit quaternions, the dot product gives cos(theta/2)
    quat_diff = torch.abs(torch.sum(object_rot * goal_rot, dim=-1))
    # Clamp to avoid numerical issues
    quat_diff = torch.clamp(quat_diff, 0.0, 1.0)
    # Convert to angle error (0 to pi/2 range)
    orientation_error = 1.0 - quat_diff
    
    # Linear velocity penalty - we want the object to stay in place while rotating
    linvel_penalty = torch.norm(object_linvel, dim=-1)
    
    # Angular velocity - some spinning is needed, but excessive spinning might be unstable
    angvel_magnitude = torch.norm(object_angvel, dim=-1)
    
    # Temperature parameters for reward shaping
    orientation_temp = 10.0
    linvel_temp = 2.0
    angvel_temp = 0.5
    
    # Reward components
    orientation_reward = torch.exp(-orientation_temp * orientation_error)
    linvel_reward = torch.exp(-linvel_temp * linvel_penalty)
    angvel_reward = torch.exp(-angvel_temp * torch.abs(angvel_magnitude - 2.0))  # Target moderate angular velocity
    
    # Total reward
    total_reward = orientation_reward * linvel_reward * angvel_reward
    
    reward_components = {
        "orientation_reward": orientation_reward,
        "linvel_reward": linvel_reward,
        "angvel_reward": angvel_reward
    }
    
    return total_reward, reward_components
