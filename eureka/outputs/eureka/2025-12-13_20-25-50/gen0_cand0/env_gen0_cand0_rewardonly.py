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
    rel_quat = torch.zeros((object_rot.shape[0], 4), device=object_rot.device)
    rel_quat[:, 0] = object_rot[:, 0] * goal_rot[:, 0] + object_rot[:, 1] * goal_rot[:, 1] + object_rot[:, 2] * goal_rot[:, 2] + object_rot[:, 3] * goal_rot[:, 3]
    rel_quat[:, 1] = object_rot[:, 0] * goal_rot[:, 1] - object_rot[:, 1] * goal_rot[:, 0] - object_rot[:, 2] * goal_rot[:, 3] + object_rot[:, 3] * goal_rot[:, 2]
    rel_quat[:, 2] = object_rot[:, 0] * goal_rot[:, 2] + object_rot[:, 1] * goal_rot[:, 3] - object_rot[:, 2] * goal_rot[:, 0] - object_rot[:, 3] * goal_rot[:, 1]
    rel_quat[:, 3] = object_rot[:, 0] * goal_rot[:, 3] - object_rot[:, 1] * goal_rot[:, 2] + object_rot[:, 2] * goal_rot[:, 1] - object_rot[:, 3] * goal_rot[:, 0]
    
    # The angle of the relative rotation is related to the scalar part (w)
    # Distance in SO(3): d = 1 - |<q1, q2>|; but we use exponential form for smoothness
    orientation_error = 1.0 - torch.abs(rel_quat[:, 0])
    
    # Temperature for orientation reward
    orientation_temp = 2.0
    orientation_reward = torch.exp(-orientation_temp * orientation_error)

    # Encourage appropriate angular velocity for spinning (not too slow, not too fast)
    angvel_norm = torch.norm(object_angvel, dim=-1)
    target_angvel = 5.0  # target magnitude for stable spinning
    angvel_error = torch.abs(angvel_norm - target_angvel)
    angvel_temp = 0.5
    angvel_reward = torch.exp(-angvel_temp * angvel_error)

    # Action regularization: penalize large actions
    action_penalty = torch.sum(actions ** 2, dim=-1)
    action_temp = 0.01
    action_reward = -action_temp * action_penalty

    # Joint velocity regularization: discourage overly fast joint movements
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