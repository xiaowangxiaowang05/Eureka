@torch.jit.script
def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    shadow_hand_dof_vel: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error using quaternion distance
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    rot_error = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:], dim=-1), min=0.0, max=1.0))  # in [0, pi]

    # Orientation reward: higher when closer to target orientation
    rot_temp = 1.0
    rot_reward = torch.exp(-rot_temp * rot_error)

    # Angular velocity regularization: penalize excessive spinning beyond what's needed
    angvel_norm = torch.norm(object_angvel, dim=-1)
    angvel_temp = 0.5
    angvel_penalty = torch.exp(-angvel_temp * angvel_norm)

    # Action smoothness / energy penalty
    action_penalty = torch.sum(actions ** 2, dim=-1)
    action_temp = 0.01
    action_reward = torch.exp(-action_temp * action_penalty)

    # Joint velocity regularization to avoid jitter
    joint_vel_norm = torch.sum(shadow_hand_dof_vel ** 2, dim=-1)
    joint_vel_temp = 0.001
    joint_vel_reward = torch.exp(-joint_vel_temp * joint_vel_norm)

    # Combine rewards
    total_reward = (
        2.0 * rot_reward +
        0.5 * angvel_penalty +
        0.3 * action_reward +
        0.2 * joint_vel_reward
    )

    reward_components = {
        "rot_reward": rot_reward,
        "angvel_penalty": angvel_penalty,
        "action_reward": action_reward,
        "joint_vel_reward": joint_vel_reward
    }

    return total_reward, reward_components