@torch.jit.script
def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    actions: torch.Tensor,
    progress_buf: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute relative rotation between object and goal
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    
    # Compute rotation error as angle of relative rotation (geodesic distance on SO(3))
    rot_error = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:], dim=-1), max=1.0))
    
    # Temperature-scaled orientation reward: higher when error is small
    orientation_temp: float = 5.0
    orientation_reward = torch.exp(-orientation_temp * rot_error)
    
    # Small action penalty to encourage efficiency (not too strong to avoid freezing)
    action_penalty = torch.sum(actions ** 2, dim=-1)
    action_temp: float = 0.01
    action_reward = -action_temp * action_penalty
    
    # Total reward
    reward = orientation_reward + action_reward
    
    reward_components = {
        "orientation_reward": orientation_reward,
        "action_reward": action_reward,
        "rot_error": rot_error
    }
    
    return reward, reward_components