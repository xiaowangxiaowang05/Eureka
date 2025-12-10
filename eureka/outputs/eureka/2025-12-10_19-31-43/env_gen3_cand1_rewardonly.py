@torch.jit.script
def compute_reward(
    object_pos: torch.Tensor,
    object_rot: torch.Tensor,
    goal_pos: torch.Tensor,
    goal_rot: torch.Tensor,
    actions: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute position distance to encourage staying near the goal position (prevents freezing far away)
    pos_dist = torch.norm(object_pos - goal_pos, p=2, dim=-1)
    dist_temp: float = 2.0
    dist_reward = torch.exp(-dist_temp * pos_dist)

    # Compute rotation error: geodesic distance on SO(3)
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    rot_error = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:], dim=-1), max=1.0))
    
    # Exponential reward for orientation alignment
    rot_temp: float = 5.0
    rot_reward = torch.exp(-rot_temp * rot_error)
    
    # Action regularization: mild penalty to reduce jitter but allow movement
    action_penalty = torch.sum(actions ** 2, dim=-1)
    action_temp: float = 0.001
    action_reg = -action_temp * action_penalty

    # Total reward combines proximity, orientation, and smoothness
    reward = dist_reward + rot_reward + action_reg

    reward_components = {
        "dist_reward": dist_reward,
        "rot_reward": rot_reward,
        "action_reg": action_reg,
        "rot_error": rot_error,
        "pos_dist": pos_dist
    }
    
    return reward, reward_components