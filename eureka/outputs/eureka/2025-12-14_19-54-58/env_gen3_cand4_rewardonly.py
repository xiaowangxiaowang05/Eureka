@torch.jit.script
def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    progress_buf: torch.Tensor,
    max_episode_length: float
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error using quaternion distance
    rel_quat = quat_mul(object_rot, quat_conjugate(goal_rot))
    angle_error = 2.0 * torch.asin(torch.clamp(torch.norm(rel_quat[:, 0:3], dim=-1), min=0.0, max=1.0))
    
    orientation_temp = 1.0
    orientation_reward = torch.exp(-orientation_temp * angle_objective)

    # Spin maintenance around object Z-axis
    spin_axis = torch.zeros_like(object_angvel)
    spin_axis[:, 2] = 1.0
    spin_proj = torch.sum(object_angvel * spin_axis, dim=-1)
    spin_temp = 0.5
    spin_reward = torch.exp(-spin_temp * torch.abs(spin_proj - 5.0))

    # Action regularization
    action_penalty = -torch.sum(actions ** 2, dim=-1) * 0.01

    # Timeout penalty
    time_factor = (progress_buf / max_episode_length).float()
    timeout_penalty = -time_factor * (1.0 - orientation_reward) * 0.1

    # NOTE: The following penalties require additional state not in current signature.
    # However, per instructions, we must use ONLY the provided inputs.
    # Therefore, we approximate smoothness via action differences using only current actions,
    # but without previous actions, true smoothness can't be computed.
    # Instead, we strengthen action penalty and add implicit constraints via existing terms.
    
    # Since we cannot access dof_pos, dof_vel, or prev_actions from signature,
    # we rely on stronger action regularization and reduce reliance on uncomputable terms.
    # The visual feedback suggests excessive high-frequency actions, so we increase action penalty weight.
    action_penalty = -torch.sum(actions ** 2, dim=-1) * 0.05

    total_reward = (
        orientation_reward * 2.0 +
        spin_reward * 1.0 +
        action_penalty +
        timeout_penalty
    )
    
    reward_components = {
        "orientation_reward": orientation_reward,
        "spin_reward": spin_reward,
        "action_penalty": action_penalty,
        "timeout_penalty": timeout_penalty
    }
    
    return total_reward, reward_components