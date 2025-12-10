@torch.jit.script
def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    actions: torch.Tensor,
    progress_buf: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute relative rotation between object and goal
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    
    # Compute geodesic distance on SO(3): angle of rotation error
    # ||quat_diff.imag|| = sin(theta/2), so theta = 2 * arcsin(||imag||)
    rot_error = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:], p=2, dim=-1), max=1.0))
    
    # Orientation reward: higher when rotation error is small
    # Use exponential with temperature for smooth gradient
    orientation_temp = 5.0
    orientation_reward = torch.exp(-orientation_temp * rot_error)
    
    # Small action penalty to encourage efficiency (not too strong to avoid freezing)
    action_penalty = torch.sum(actions ** 2, dim=-1)
    action_penalty_scale = 0.001
    
    # Total reward
    reward = orientation_reward - action_penalty_scale * action_penalty
    
    reward_components = {
        "orientation_reward": orientation_reward,
        "action_penalty": action_penalty,
        "rot_error": rot_error
    }
    
    return reward, reward_components