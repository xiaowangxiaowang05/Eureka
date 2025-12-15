@torch.jit.script
def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    progress_buf: torch.Tensor,
    max_episode_length: float
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Orientation alignment reward (dense)
    rel_quat = quat_mul(object_rot, quat_conjugate(goal_rot))
    angle_error = 2.0 * torch.asin(torch.clamp(torch.norm(rel_quat[:, 0:3], dim=-1), min=0.0, max=1.0))
    orientation_temp = 2.0  # sharper signal near target
    orientation_reward = torch.exp(-orientation_temp * angle_error)

    # Spin maintenance: encourage constant rotation around object's local Z-axis
    # Project angular velocity onto object's current up direction (Z in object frame)
    object_up = quat_apply(object_rot, torch.tensor([0.0, 0.0, 1.0], device=object_rot.device).repeat(object_rot.shape[0], 1))
    spin_proj = torch.sum(object_angvel * object_up, dim=-1)
    # Target moderate spin (not too fast to avoid instability, not zero)
    desired_spin = 3.0
    spin_error = torch.abs(spin_proj - desired_spin)
    spin_temp = 0.8
    spin_reward = torch.exp(-spin_temp * spin_error)

    # Stronger default pose penalty: discourage extreme joint configurations (hyperextension)
    default_pose_penalty = -torch.sum(actions ** 2, dim=-1) * 0.05  # increased from 0.02

    # Action magnitude penalty (energy-like cost)
    action_penalty = -torch.sum(actions ** 2, dim=-1) * 0.015

    # Torque-effort proxy: L1 penalty to reduce high-frequency actuation
    torque_penalty = -torch.sum(torch.abs(actions), dim=-1) * 0.008

    # Time-based timeout penalty scaled by task progress
    time_ratio = (progress_buf / max_episode_length).float()
    timeout_penalty = -time_ratio * (1.0 - orientation_reward) * 0.15

    total_reward = (
        orientation_reward * 2.5 +
        spin_reward * 1.2 +
        default_pose_penalty +
        action_penalty +
        torque_penalty +
        timeout_penalty
    )

    reward_components = {
        "orientation_reward": orientation_reward,
        "spin_reward": spin_reward,
        "default_pose_penalty": default_pose_penalty,
        "action_penalty": action_penalty,
        "torque_penalty": torque_penalty,
        "timeout_penalty": timeout_penalty
    }

    return total_reward, reward_components