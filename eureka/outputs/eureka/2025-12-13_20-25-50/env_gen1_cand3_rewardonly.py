@torch.jit.script
def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    shadow_hand_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute rotation error using quaternion angle difference (dense, smooth)
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))  # [0, pi]

    # Dense orientation reward: higher when rotation error is small
    rot_reward_temp = 1.5
    rot_reward = torch.exp(-rot_reward_temp * rot_dist)

    # Encourage ANY controlled motion (do not enforce specific spin speed)
    angvel_norm = torch.norm(object_angvel, dim=-1)
    motion_reward_temp = 0.5
    motion_reward = torch.exp(-motion_reward_temp * angvel_norm)  # prefer low spin? No—flip logic!
    # Actually: we want SOME spin to avoid static grip → reward moderate angvel
    target_angvel = 1.0
    angvel_error = torch.abs(angvel_norm - target_angvel)
    angvel_reward_temp = 1.0
    angvel_reward = torch.exp(-angvel_reward_temp * angvel_error)

    # Action regularization (reduce weight to avoid freezing)
    action_penalty = torch.sum(actions ** 2, dim=-1)
    action_penalty_weight = 0.001
    action_reward = -action_penalty_weight * action_penalty

    # Joint velocity penalty (greatly reduced weight)
    joint_vel_penalty = torch.sum(shadow_hand_dof_vel ** 2, dim=-1)
    joint_vel_penalty_weight = 0.0001
    joint_vel_reward = -joint_vel_penalty_weight * joint_vel_penalty

    # Combine rewards
    total_reward = (
        rot_reward +
        0.8 * angvel_reward +
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