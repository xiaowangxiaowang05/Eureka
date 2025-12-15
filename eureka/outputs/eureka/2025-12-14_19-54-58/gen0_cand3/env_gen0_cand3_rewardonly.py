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
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    # Extract angle-axis representation; angle is 2*acos(w)
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:], dim=-1), min=0.0, max=1.0))
    
    # Orientation reward: encourage alignment with goal orientation
    orientation_temp = 1.0
    orientation_reward = torch.exp(-orientation_temp * rot_dist)

    # Encourage some angular velocity to keep spinning (prevent static alignment only)
    angvel_norm = torch.norm(object_angvel, dim=-1)
    target_angvel = 2.0  # desired magnitude of angular velocity
    angvel_error = torch.abs(angvel_norm - target_angvel)
    angvel_temp = 0.5
    angvel_reward = torch.exp(-angvel_temp * angvel_error)

    # Action regularization: penalize large actions
    action_penalty = -torch.sum(actions ** 2, dim=-1) * 0.01

    # Joint velocity regularization: discourage excessive joint motion
    joint_vel_penalty = -torch.sum(shadow_hand_dof_vel ** 2, dim=-1) * 0.001

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