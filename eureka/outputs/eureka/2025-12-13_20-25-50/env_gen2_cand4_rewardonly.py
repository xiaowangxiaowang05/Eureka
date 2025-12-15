@torch.jit.script
def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    shadow_hand_dof_vel: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error as geodesic distance on SO(3)
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    rot_error = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, :3], p=2, dim=-1), max=1.0))
    
    # Dense reward for orientation: higher when closer (no saturation via exp)
    rot_reward = 1.0 / (rot_error + 0.1)

    # Encourage sustained spinning: penalize zero angular velocity
    angvel_mag = torch.norm(object_angvel, dim=-1)
    min_spin = 1.0  # minimum useful spin magnitude
    angvel_bonus = torch.clamp(angvel_mag - min_spin, min=0.0)
    angvel_reward = angvel_bonus * 0.5

    # Light action penalty for energy efficiency
    action_penalty = torch.sum(actions ** 2, dim=-1)
    action_reward = -0.01 * action_penalty

    # Very light joint velocity penalty for smoothness (prevent freezing)
    joint_vel_penalty = torch.sum(shadow_hand_dof_vel ** 2, dim=-1)
    joint_vel_reward = -0.0005 * joint_vel_penalty

    # Bonus for making progress toward goal orientation
    progress_bonus = torch.clamp(rot_reward - 5.0, min=0.0) * 0.2

    total_reward = (
        rot_reward +
        angvel_reward +
        action_reward +
        joint_vel_reward +
        progress_bonus
    )

    reward_components = {
        "rot_reward": rot_reward,
        "angvel_reward": angvel_reward,
        "action_reward": action_reward,
        "joint_vel_reward": joint_vel_reward,
        "progress_bonus": progress_bonus
    }

    return total_reward, reward_components

# Reuse helper functions from parent if needed (already defined in environment)