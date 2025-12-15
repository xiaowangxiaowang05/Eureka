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
    rel_quat = torch.linalg.norm(object_rot - goal_rot, dim=-1)
    # Alternative: use angle-axis distance via quaternion inner product
    quat_inner = torch.sum(object_rot * goal_rot, dim=-1)
    rot_error = 1.0 - torch.abs(quat_inner)  # ranges from 0 (aligned) to 1 (opposite)

    # Orientation reward: encourage alignment with goal orientation
    rot_reward_temp = 2.0
    rot_reward = torch.exp(-rot_reward_temp * rot_error)

    # Encourage maintaining some angular velocity to keep spinning (prevent static grip)
    angvel_norm = torch.norm(object_angvel, dim=-1)
    target_angvel_norm = 2.0  # desired magnitude of angular velocity
    angvel_error = torch.abs(angvel_norm - target_angvel_norm)
    angvel_reward_temp = 1.0
    angvel_reward = torch.exp(-angvel_reward_temp * angvel_error)

    # Action regularization: penalize large actions
    action_penalty = torch.sum(actions ** 2, dim=-1)
    action_penalty_weight = 0.01
    action_reward = -action_penalty_weight * action_penalty

    # Joint velocity regularization: discourage excessive joint velocities
    joint_vel_penalty = torch.sum(shadow_hand_dof_vel ** 2, dim=-1)
    joint_vel_penalty_weight = 0.001
    joint_vel_reward = -joint_vel_penalty_weight * joint_vel_penalty

    # Combine rewards
    total_reward = (
        rot_reward +
        0.5 * angvel_reward +
        action_reward +
        joint_vel_reward
    )

    reward_components = {
        "rot_reward": rot_reward,
        "angvel_reward": angvel_reward,
        "action_reward": action_reward,
        "joint_vel_reward": joint_vel_reward
    }

    return total_reward, reward_components