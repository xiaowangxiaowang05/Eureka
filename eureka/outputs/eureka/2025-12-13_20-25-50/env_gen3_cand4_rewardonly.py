@torch.jit.script
def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    shadow_hand_dof_vel: torch.Tensor,
    fingertip_pos: torch.Tensor,
    object_pos: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation alignment reward
    rel_quat = quat_mul(object_rot, quat_conjugate(goal_rot))
    orientation_error = 1.0 - torch.abs(rel_quat[:, 0])
    orientation_temp = 2.0
    orientation_reward = torch.exp(-orientation_temp * orientation_error)

    # Encourage controlled spinning
    angvel_norm = torch.norm(object_angvel, dim=-1)
    target_angvel = 4.0
    angvel_error = torch.abs(angvel_norm - target_angvel)
    angvel_temp = 0.5
    angvel_reward = torch.exp(-angvel_temp * angvel_error)

    # Action regularization
    action_penalty = torch.sum(actions ** 2, dim=-1)
    action_temp = 0.01
    action_reward = -action_temp * action_penalty

    # Joint velocity regularization
    joint_vel_penalty = torch.sum(shadow_hand_dof_vel ** 2, dim=-1)
    joint_vel_temp = 0.001
    joint_vel_reward = -joint_vel_temp * joint_vel_penalty

    # Object drop penalty: distance from hand center (average fingertip)
    hand_center = torch.mean(fingertip_pos, dim=1)
    obj_hand_dist = torch.norm(object_pos - hand_center, dim=-1)
    drop_temp = 5.0
    drop_penalty = -torch.exp(drop_temp * (obj_hand_dist - 0.2))  # strong penalty if >0.2m away

    # Combine rewards
    total_reward = (
        orientation_reward +
        0.5 * angvel_reward +
        action_reward +
        joint_vel_reward +
        drop_penalty
    )

    reward_components = {
        "orientation_reward": orientation_reward,
        "angvel_reward": angvel_reward,
        "action_reward": action_reward,
        "joint_vel_reward": joint_vel_reward,
        "drop_penalty": drop_penalty
    }

    return total_reward, reward_components