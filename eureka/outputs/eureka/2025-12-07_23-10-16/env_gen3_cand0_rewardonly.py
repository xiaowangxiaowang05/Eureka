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

    # --- Linear velocity penalty to keep object stable ---
    linvel_norm = torch.norm(object_linvel, dim=-1)
    linvel_temp = 2.0
    linvel_reward = torch.exp(-linvel_temp * linvel_norm)

    # --- Enhanced adaptive angular velocity reward ---
    angvel_norm = torch.norm(object_angvel, dim=-1)
    
    # More sensitive proximity threshold to start fine control earlier
    proximity_threshold = 0.15
    close_to_target = orientation_error <= proximity_threshold

    # Stronger penalty when close to target to prevent overshooting and wobbling
    angvel_penalty_close = torch.exp(-8.0 * angvel_norm)

    # When far: encourage moderate spinning 
    angvel_reward_far = torch.exp(-0.5 * torch.abs(angvel_norm - 2.0))

    # Combine based on proximity to target
    angvel_reward = torch.where(close_to_target, angvel_penalty_close, angvel_reward_far)

    # --- Additional fine control: extra penalty for any motion when very close ---
    very_close_threshold = 0.05
    very_close_to_target = orientation_error <= very_close_threshold
    fine_control_penalty = torch.where(
        very_close_to_target,
        torch.exp(-10.0 * angvel_norm) * torch.exp(-5.0 * linvel_norm),
        torch.ones_like(angvel_norm)
    )

    # --- Total reward: multiplicative for stability + orientation, additive for adaptive angvel ---
    total_reward = orientation_reward * linvel_reward * fine_control_penalty + 0.2 * angvel_reward

    reward_components = {
        "orientation_reward": orientation_reward,
        "linvel_reward": linvel_reward,
        "angvel_reward": angvel_reward,
        "fine_control_penalty": fine_control_penalty
    }

    return total_reward, reward_components
