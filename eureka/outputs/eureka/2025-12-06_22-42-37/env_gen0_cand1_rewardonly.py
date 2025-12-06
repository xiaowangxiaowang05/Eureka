@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, object_angvel: torch.Tensor, object_linvel: torch.Tensor, progress_buf: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate orientation error using quaternion distance
    # The distance between two quaternions q1 and q2 is: 1 - |q1 · q2|
    # Since quaternions are normalized, dot product gives cos(theta/2)
    quat_diff = torch.abs(torch.sum(object_rot * goal_rot, dim=1))
    orientation_error = 1.0 - quat_diff
    
    # Reward for good orientation alignment (higher when error is smaller)
    orientation_reward_temp = 1.0
    orientation_reward = torch.exp(-orientation_error / orientation_reward_temp)
    
    # Penalize excessive linear velocity (keep object stable in position)
    linvel_penalty_temp = 0.5
    linvel_magnitude = torch.norm(object_linvel, dim=1)
    linvel_penalty = torch.exp(-linvel_magnitude / linvel_penalty_temp)
    
    # Encourage appropriate angular velocity for spinning
    # We want some angular velocity but not too much chaotic spinning
    angvel_magnitude = torch.norm(object_angvel, dim=1)
    angvel_reward_temp = 2.0
    # Optimal angular velocity around 2-4 rad/s for controlled spinning
    optimal_angvel = 3.0
    angvel_error = torch.abs(angvel_magnitude - optimal_angvel)
    angvel_reward = torch.exp(-angvel_error / angvel_reward_temp)
    
    # Combine rewards with weights
    total_reward = (
        2.0 * orientation_reward + 
        1.0 * angvel_reward + 
        0.5 * linvel_penalty
    )
    
    reward_components = {
        "orientation_reward": orientation_reward,
        "angvel_reward": angvel_reward,
        "linvel_penalty": linvel_penalty,
        "orientation_error": orientation_error,
        "angvel_magnitude": angvel_magnitude,
        "linvel_magnitude": linvel_magnitude
    }
    
    return total_reward, reward_components
