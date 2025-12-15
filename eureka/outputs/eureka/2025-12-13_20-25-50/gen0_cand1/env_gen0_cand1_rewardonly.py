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
    rot_dist = 1.0 - torch.abs(rel_quat[:, 0])  # [0, 1]
    
    # Angular velocity alignment: encourage spinning in correct axis/direction
    # For general orientation matching, we don't necessarily want high angvel,
    # but we do want the object to be stable near target orientation
    angvel_norm = torch.norm(object_angvel, dim=-1)
    angvel_penalty = torch.exp(-1.0 * angvel_norm)  # Prefer low angular velocity when aligned
    
    # Combine orientation and stability
    orientation_reward_temp = 2.0
    orientation_reward = torch.exp(-orientation_reward_temp * rot_dist)
    
    stability_reward_temp = 1.0
    stability_reward = torch.exp(-stability_reward_temp * angvel_norm)
    
    final_orientation_reward = orientation_reward * stability_reward

    # Action regularization: penalize large actions
    action_penalty_temp = 0.05
    action_penalty = torch.sum(actions ** 2, dim=-1)
    action_reward = torch.exp(-action_penalty_temp * action_penalty)
    
    # Joint velocity penalty to avoid jitter
    joint_vel_penalty_temp = 0.01
    joint_vel_penalty = torch.sum(shadow_hand_dof_vel ** 2, dim=-1)
    joint_vel_reward = torch.exp(-joint_vel_penalty_temp * joint_vel_penalty)

    # Total reward components
    total_reward = (
        2.0 * final_orientation_reward +
        0.5 * action_reward +
        0.3 * joint_vel_reward
    )
    
    reward_components = {
        "final_orientation_reward": final_orientation_reward,
        "action_reward": action_reward,
        "joint_vel_reward": joint_vel_reward,
        "rot_dist": rot_dist,
        "angvel_norm": angvel_norm
    }
    
    return total_reward, reward_components


# Helper functions required for TorchScript compatibility
@torch.jit.script
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([-q[:, :3], q[:, 3:4]], dim=-1)

@torch.jit.script
def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape == b.shape
    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([x, y, z, w], dim=-1)