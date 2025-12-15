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
    # The w component of the relative quaternion relates to the angle error
    # error_angle = 2 * acos(|w|), so we use |w| as a proxy for alignment
    rot_error = 1.0 - torch.abs(rel_quat[:, 3])  # w component is at index 3
    
    # Temperature parameter for orientation reward shaping
    rot_reward_temp = 2.0
    rot_reward = torch.exp(-rot_reward_temp * rot_error)
    
    # Encourage maintaining some angular velocity to keep spinning (prevent stalling)
    angvel_norm = torch.norm(object_angvel, dim=-1)
    angvel_target = 2.0  # target angular speed magnitude
    angvel_error = torch.abs(angvel_norm - angvel_target)
    angvel_reward_temp = 0.5
    angvel_reward = torch.exp(-angvel_reward_temp * angvel_error)
    
    # Regularization penalties
    # Action smoothness penalty (difference between consecutive actions is handled externally;
    # here we penalize large actions)
    action_penalty = torch.sum(actions ** 2, dim=-1)
    action_penalty_weight = 0.01
    
    # Joint velocity penalty to discourage excessive motion
    joint_vel_penalty = torch.sum(shadow_hand_dof_vel ** 2, dim=-1)
    joint_vel_penalty_weight = 0.001
    
    # Total reward components
    reward = (
        rot_reward 
        + 0.5 * angvel_reward 
        - action_penalty_weight * action_penalty 
        - joint_vel_penalty_weight * joint_vel_penalty
    )
    
    reward_components = {
        "rot_reward": rot_reward,
        "angvel_reward": angvel_reward,
        "action_penalty": action_penalty_weight * action_penalty,
        "joint_vel_penalty": joint_vel_penalty_weight * joint_vel_penalty
    }
    
    return reward, reward_components


# Helper functions required for torch.jit compatibility
@torch.jit.script
def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[-1] == 4
    assert b.shape[-1] == 4

    x1, y1, z1, w1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    x2, y2, z2, w2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

    return torch.stack([x, y, z, w], dim=-1)

@torch.jit.script
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([-q[..., :3], q[..., 3:4]], dim=-1)