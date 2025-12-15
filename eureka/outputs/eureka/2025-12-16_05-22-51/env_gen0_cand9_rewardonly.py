@torch.jit.script
def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    shadow_hand_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error using quaternion distance
    # quat_mul(q1, quat_conjugate(q2)) gives relative rotation from q2 to q1
    rot_error_quat = quat_mul(object_rot, quat_conjugate(goal_rot))
    # Extract angle-axis representation; angle is 2*arccos(w)
    w = rot_error_quat[:, 3]  # scalar part
    angle_error = 2.0 * torch.acos(torch.clamp(w, -1.0 + 1e-6, 1.0 - 1e-6))
    # Normalize to [0, pi]
    angle_error = torch.abs(angle_error)

    # Orientation reward: higher when closer to target orientation
    orientation_temp = 1.0
    orientation_reward = torch.exp(-orientation_temp * angle_error)

    # Encourage appropriate angular velocity for spinning (not too slow, not too fast)
    # Target some minimal spin to prevent just holding still at correct orientation
    angvel_norm = torch.norm(object_angvel, dim=-1)
    desired_angvel = 2.0  # rad/s, a moderate spin rate
    angvel_error = torch.abs(angvel_norm - desired_angvel)
    angvel_temp = 0.5
    angvel_reward = torch.exp(-angvel_temp * angvel_error)

    # Action regularization: penalize large actions
    action_penalty = -torch.sum(actions ** 2, dim=-1) * 0.001

    # Joint velocity regularization: discourage excessive joint speeds
    joint_vel_penalty = -torch.sum(shadow_hand_dof_vel ** 2, dim=-1) * 0.0001

    # Combine rewards
    total_reward = (
        orientation_reward +
        angvel_reward +
        action_penalty +
        joint_vel_penalty
    )

    reward_components = {
        "orientation_reward": orientation_reward,
        "angvel_reward": angvel_reward,
        "action_penalty": action_penalty,
        "joint_vel_penalty": joint_vel_penalty
    }

    return total_reward, reward_components