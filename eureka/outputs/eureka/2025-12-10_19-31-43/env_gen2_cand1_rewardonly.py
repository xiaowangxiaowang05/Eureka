@torch.jit.script
def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    actions: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute relative rotation between object and goal: q_diff = object_rot * conjugate(goal_rot)
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    
    # Compute rotation error as the angle of the relative rotation (geodesic distance on SO(3))
    # ||vec(q)|| = sin(theta/2), so theta = 2 * arcsin(||vec(q)||)
    rot_error = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:], dim=-1), max=1.0))
    
    # Use exponential reward based on rotation error (higher when aligned)
    rot_temp: float = 5.0
    rot_reward = torch.exp(-rot_temp * rot_error)
    
    # Small penalty on action magnitude to discourage jitter while allowing motion
    action_penalty = torch.sum(actions ** 2, dim=-1)
    action_temp: float = 0.01
    action_reg = -action_temp * action_penalty
    
    # Total reward
    reward = rot_reward + action_reg
    
    reward_components = {
        "rot_reward": rot_reward,
        "action_reg": action_reg,
        "rot_error": rot_error
    }
    
    return reward, reward_components