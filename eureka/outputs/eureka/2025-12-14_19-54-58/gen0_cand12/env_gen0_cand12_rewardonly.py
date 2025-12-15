def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor, 
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    object_linvel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Orientation alignment reward using quaternion distance
    # quat_mul(object_rot, quat_conjugate(goal_rot)) gives relative rotation
    # The scalar part (w component) indicates alignment: 1.0 = perfect alignment
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    orientation_alignment = quat_diff[:, 3]  # w component
    orientation_reward = orientation_alignment
    
    # Angular velocity regularization - we want controlled spinning, not excessive speed
    # But some angular velocity is needed for dynamic manipulation
    angvel_norm = torch.norm(object_angvel, dim=-1)
    angvel_temp = 2.0
    angvel_reward = torch.exp(-angvel_norm / angvel_temp)
    
    # Linear velocity penalty - object should stay relatively stable in position while spinning
    linvel_norm = torch.norm(object_linvel, dim=-1)
    linvel_temp = 1.0
    linvel_penalty = torch.exp(-linvel_norm / linvel_temp)
    
    # Action regularization to encourage energy efficiency
    action_norm = torch.norm(actions, dim=-1)
    action_temp = 0.1
    action_penalty = torch.exp(-action_norm / action_temp)
    
    # Combine rewards with appropriate weights
    orientation_weight = 2.0
    angvel_weight = 0.5
    linvel_weight = 0.5
    action_weight = 0.1
    
    total_reward = (
        orientation_weight * orientation_reward +
        angvel_weight * angvel_reward +
        linvel_weight * linvel_penalty +
        action_weight * action_penalty
    )
    
    reward_components = {
        "orientation_reward": orientation_reward,
        "angvel_reward": angvel_reward,
        "linvel_penalty": linvel_penalty,
        "action_penalty": action_penalty
    }
    
    return total_reward, reward_components

# Helper functions needed for torch.jit compatibility
@torch.jit.script
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([-q[:, :3], q[:, 3:4]], dim=-1)

@torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return torch.stack([x, y, z, w], dim=-1)