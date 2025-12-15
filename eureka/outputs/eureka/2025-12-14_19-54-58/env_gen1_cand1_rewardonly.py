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
    
    # Dense orientation reward
    orientation_temp = 1.0
    orientation_reward = torch.exp(-orientation_temp * angle_error)
    
    # Spin maintenance around object Z-axis (target ~5 rad/s)
    spin_axis = torch.zeros_like(object_angvel)
    spin_axis[:, 2] = 1.0
    spin_proj = torch.sum(object_angvel * spin_axis, dim=-1)
    spin_temp = 0.5
    spin_reward = torch.exp(-spin_temp * torch.abs(spin_proj - 5.0))
    
    # Action regularization (energy minimization)
    action_penalty = -torch.sum(actions ** 2, dim=-1) * 0.01
    
    # Timeout penalty to encourage faster completion
    time_factor = (progress_buf / max_episode_length).float()
    timeout_penalty = -time_factor * (1.0 - orientation_reward) * 0.1
    
    # NEW: Joint acceleration penalty (requires velocity history, but we approximate with action differences if prev_actions available)
    # Since prev_actions isn't in inputs, we skip true acceleration; instead, we penalize large action changes via smoothing
    # But since we don't have prev_actions here, we use a proxy: high-frequency jitter often correlates with large action magnitudes
    # We already have action penalty, so we enhance it with a stronger default pose deviation penalty
    
    # NEW: Deviation from default joint pose (shadow_hand_default_dof_pos is [0]*20, so we assume default is zero)
    # However, since we don't have access to dof_pos in this function signature, we cannot compute this directly.
    # Given the constraints of the function signature (only provided inputs), we must work within them.
    # The visual feedback suggests high-frequency jitter, which manifests as high action derivatives.
    # Without prev_actions, the best proxy is to increase the action penalty weight on high-frequency components,
    # but TorchScript doesn't support Fourier transforms. Instead, we note that jitter often requires large actions,
    # so we keep the action penalty but slightly increase its weight to discourage extreme actuation.
    
    # After analysis: Since the function signature doesn't include previous actions or joint positions,
    # we cannot implement acceleration or default pose penalties directly.
    # However, the original action penalty is too weak (-0.01 scale); increasing it will help reduce jitter.
    # Also, we add a penalty for excessive angular velocity magnitude to prevent violent spinning.
    angvel_penalty = -torch.norm(object_angvel, dim=-1) * 0.05
    
    # Combine rewards with adjusted weights
    total_reward = (
        orientation_reward * 2.0 +
        spin_reward * 1.0 +
        action_penalty * 2.0 +  # increased weight to combat jitter
        angvel_penalty +
        timeout_penalty
    )
    
    reward_components = {
        "orientation_reward": orientation_reward,
        "spin_reward": spin_reward,
        "action_penalty": action_penalty * 2.0,
        "angvel_penalty": angvel_penalty,
        "timeout_penalty": timeout_penalty
    }
    
    return total_reward, reward_components