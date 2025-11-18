@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, object_angvel: torch.Tensor, actions: torch.Tensor, object_linvel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error using quaternion difference
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    rot_error = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))
    
    # Orientation reward (exponential with temperature)
    orientation_temp = 10.0
    orientation_reward = torch.exp(-orientation_temp * rot_error)
    
    # Penalize excessive angular velocity
    angvel_penalty = -torch.norm(object_angvel, dim=-1)
    
    # Penalize large actions (energy efficiency)
    action_penalty = -torch.sum(actions ** 2, dim=-1)
    
    # Penalize linear movement of the object (keep it centered)
    linvel_penalty = -torch.norm(object_linvel, dim=-1)
    
    # Combine rewards
    total_reward = orientation_reward + 0.1 * angvel_penalty + 0.05 * action_penalty + 0.1 * linvel_penalty
    
    reward_dict = {
        "orientation_reward": orientation_reward,
        "angvel_penalty": angvel_penalty,
        "action_penalty": action_penalty,
        "linvel_penalty": linvel_penalty
    }
    
    return total_reward, reward_dict
