def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    shadow_hand_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error using quaternion distance
    # quat_mul(q1, quat_conjugate(q2)) gives relative rotation from q2 to q1
    rel_quat = quat_mul(object_rot, quat_conjugate(goal_rot))
    # Extract w component; closer to 1 means more aligned
    rot_dist = 1.0 - torch.abs(rel_quat[:, 3])  # [0, 1]
    
    # Angular velocity alignment: encourage spinning around correct axis
    # For general orientation tasks, we may not care about specific spin axis,
    # but we do want smooth motion and convergence
    angvel_norm = torch.norm(object_angvel, dim=-1)
    
    # Action regularization to reduce jitter and energy use
    action_penalty = torch.sum(actions ** 2, dim=-1)
    
    # Joint velocity penalty for smoothness
    joint_vel_penalty = torch.sum(shadow_hand_dof_vel ** 2, dim=-1)

    # Temperature parameters for reward shaping
    rot_temp = 2.0
    angvel_temp = 1.0
    action_temp = 0.01
    joint_vel_temp = 0.001

    # Shaped rewards
    rot_reward = torch.exp(-rot_temp * rot_dist)
    angvel_reward = torch.exp(-angvel_temp * angvel_norm)  # Prefer low residual spin when aligned
    
    action_reg = -action_temp * action_penalty
    joint_vel_reg = -joint_vel_temp * joint_vel_penalty

    total_reward = rot_reward + angvel_reward + action_reg + joint_vel_reg

    reward_components = {
        "rot_reward": rot_reward,
        "angvel_reward": angvel_reward,
        "action_reg": action_reg,
        "joint_vel_reg": joint_vel_reg
    }

    return total_reward, reward_components

# Helper functions required for TorchScript compatibility
@torch.jit.script
def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[-1] == 4
    assert b.shape[-1] == 4

    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=-1)

@torch.jit.script
def quat_conjugate(a: torch.Tensor) -> torch.Tensor:
    return torch.cat([a[..., 0:1], -a[..., 1:]], dim=-1)