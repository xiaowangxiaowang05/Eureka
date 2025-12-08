@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, object_angvel: torch.Tensor, progress_buf: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute quaternion distance between current and goal orientation
    # Quaternion distance: ||q1 - q2||^2, but we need to account for quaternion double cover
    # So we take min(||q1 - q2||, ||q1 + q2||)
    quat_diff = torch.abs(object_rot - goal_rot)
    quat_diff_alt = torch.abs(object_rot + goal_rot)
    quat_dist = torch.minimum(torch.sum(quat_diff * quat_diff, dim=-1), 
                             torch.sum(quat_diff_alt * quat_diff_alt, dim=-1))
    
    # Orientation reward: higher when closer to target orientation
    orientation_temp = 10.0
    orientation_reward = torch.exp(-orientation_temp * quat_dist)
    
    # Encourage some angular velocity to promote spinning motion (but not too much)
    angvel_norm = torch.norm(object_angvel, dim=-1)
    angvel_temp = 0.1
    angvel_reward = torch.exp(-angvel_temp * torch.abs(angvel_norm - 2.0))  # Target moderate spinning speed
    
    # Combine rewards
    total_reward = orientation_reward + 0.1 * angvel_reward
    
    reward_components = {
        "orientation_reward": orientation_reward,
        "angvel_reward": angvel_reward
    }
    
    return total_reward, reward_components
