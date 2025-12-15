@torch.jit.script
def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    shadow_hand_dof_vel: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error using quaternion distance
    # quat_mul(q1, quat_conjugate(q2)) gives relative rotation from q2 to q1
    rot_error_quat = quat_mul(object_rot, quat_conjugate(goal_rot))
    # Extract angle from quaternion: angle = 2 * acos(|w|)
    rot_angle = 2.0 * torch.acos(torch.clamp(torch.abs(rot_error_quat[:, 3]), min=0.0, max=1.0))
    # Normalize angle to [0, pi]
    rot_angle = torch.clamp(rot_angle, max=torch.pi)
    
    # Orientation reward: higher when closer to target orientation
    orientation_temp = 1.0
    orientation_reward = torch.exp(-orientation_temp * rot_angle)

    # Encourage maintaining some angular velocity to keep spinning (prevent stalling)
    # But not too much to avoid instability
    angvel_magnitude = torch.norm(object_angvel, dim=-1)
    target_angvel = 2.0  # target magnitude for stable spinning
    angvel_error = torch.abs(angvel_magnitude - target_angvel)
    angvel_temp = 0.5
    angvel_reward = torch.exp(-angvel_temp * angvel_error)

    # Action regularization: penalize large actions for energy efficiency
    action_penalty = torch.sum(actions ** 2, dim=-1)
    action_temp = 0.01
    action_reward = -action_temp * action_penalty

    # Joint velocity penalty: discourage excessive joint movements
    joint_vel_penalty = torch.sum(shadow_hand_dof_vel ** 2, dim=-1)
    joint_vel_temp = 0.001
    joint_vel_reward = -joint_vel_temp * joint_vel_penalty

    # Combine rewards
    total_reward = (
        orientation_reward +
        0.5 * angvel_reward +
        action_reward +
        joint_vel_reward
    )

    reward_components = {
        "orientation_reward": orientation_reward,
        "angvel_reward": angvel_reward,
        "action_reward": action_reward,
        "joint_vel_reward": joint_vel_reward
    }

    return total_reward, reward_components