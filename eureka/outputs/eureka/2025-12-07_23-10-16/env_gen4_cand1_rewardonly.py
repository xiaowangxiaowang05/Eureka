@torch.jit.script
def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    object_linvel: torch.Tensor,
    progress_buf: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # --- Orientation error using quaternion dot product ---
    quat_diff = torch.abs(torch.sum(object_rot * goal_rot, dim=-1))
    quat_diff = torch.clamp(quat_diff, 0.0, 1.0)
    orientation_error = 1.0 - quat_diff

    orientation_temp = 10.0
    orientation_reward = torch.exp(-orientation_temp * orientation_error)

    # --- Enhanced linear velocity penalty for better stability ---
    linvel_norm = torch.norm(object_linvel, dim=-1)
    linvel_temp = 3.0  # Increased from 2.0 for stronger penalty
    linvel_reward = torch.exp(-linvel_temp * linvel_norm)

    # --- Improved adaptive angular velocity reward with stronger fine control ---
    angvel_norm = torch.norm(object_angvel, dim=-1)
    
    # Use a more gradual proximity threshold with multiple zones
    very_close_threshold = 0.05  # Stricter threshold for fine control
    close_threshold = 0.15       # Broader threshold for transition
    
    very_close_to_target = orientation_error <= very_close_threshold
    close_to_target = orientation_error <= close_threshold
    
    # When very close to target: extremely strong penalty for any angular motion (prevent wobbling/overshooting)
    angvel_penalty_very_close = torch.exp(-15.0 * angvel_norm)  # Much stronger penalty
    
    # When close (but not very close) to target: strong penalty for angular motion
    angvel_penalty_close = torch.exp(-8.0 * angvel_norm)  # Stronger than original
    
    # When far from target: encourage moderate spinning toward the target
    angvel_temp_far = 0.5
    angvel_reward_far = torch.exp(-angvel_temp_far * torch.abs(angvel_norm - 2.0))
    
    # Smooth combination using multiple thresholds
    angvel_reward = torch.where(
        very_close_to_target, 
        angvel_penalty_very_close,
        torch.where(close_to_target, angvel_penalty_close, angvel_reward_far)
    )

    # --- Additional stabilization: combined motion penalty when very close ---
    total_motion_norm = linvel_norm + angvel_norm
    motion_stability_bonus = torch.where(
        very_close_to_target,
        torch.exp(-10.0 * total_motion_norm),  # Extra penalty for any residual motion
        torch.ones_like(total_motion_norm)
    )

    # --- Total reward with better balance for fine control ---
    # Reduce the weight of angvel_reward slightly and add motion stability bonus
    total_reward = orientation_reward * linvel_reward * motion_stability_bonus + 0.15 * angvel_reward

    reward_components = {
        "orientation_reward": orientation_reward,
        "linvel_reward": linvel_reward,
        "angvel_reward": angvel_reward,
        "motion_stability_bonus": motion_stability_bonus
    }

    return total_reward, reward_components
