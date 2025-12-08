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
    
    # Adaptive angular velocity reward/penalty based on proximity to target
    angvel_norm = torch.norm(object_angvel, dim=-1)
    
    # When far from target (quat_dist > threshold), encourage some motion
    # When close to target (quat_dist <= threshold), strongly penalize any angular velocity
    proximity_threshold = 0.1
    close_to_target = quat_dist <= proximity_threshold
    
    # For states close to target, penalize any angular velocity
    angvel_penalty_close = torch.exp(-5.0 * angvel_norm)  # Strong penalty for any motion when close
    
    # For states far from target, encourage moderate angular velocity (original logic)
    angvel_reward_far = torch.exp(-0.1 * torch.abs(angvel_norm - 2.0))
    
    # Combine based on proximity
    angvel_reward = torch.where(close_to_target, angvel_penalty_close, angvel_reward_far)
    
    # Combine rewards with higher weight on orientation and adaptive angvel
    total_reward = orientation_reward + 0.2 * angvel_reward
    
    reward_components = {
        "orientation_reward": orientation_reward,
        "angvel_reward": angvel_reward
    }
    
    return total_reward, reward_components
