@torch.jit.script
def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    shadow_hand_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute rotation error using quaternion difference (angle of relative rotation)
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], dim=-1), max=1.0))
    
    # Dense orientation reward: higher when closer to target orientation
    rot_reward_temp = 1.5
    rot_reward = torch.exp(-rot_reward_temp * rot_dist)

    # Remove target angular velocity bonus that discouraged natural spinning
    # Instead, lightly penalize zero spin to avoid static gripping
    angvel_norm = torch.norm(object_angvel, dim=-1)
    spin_bonus = torch.clamp(angvel_norm, min=0.0, max=1.0)  # small bonus for any spin

    # Reduce action penalty weight to avoid freezing
    action_penalty = torch.sum(actions ** 2, dim=-1)
    action_reward = -0.005 * action_penalty

    # Greatly reduce joint velocity penalty to allow movement
    joint_vel_penalty = torch.sum(shadow_hand_dof_vel ** 2, dim=-1)
    joint_vel_reward = -0.0001 * joint_vel_penalty

    total_reward = (
        rot_reward +
        0.2 * spin_bonus +
        action_reward +
        joint_vel_reward
    )

    reward_components = {
        "rot_reward": rot_reward,
        "spin_bonus": spin_bonus,
        "action_reward": action_reward,
        "joint_vel_reward": joint_vel_reward
    }

    return total_reward, reward_components