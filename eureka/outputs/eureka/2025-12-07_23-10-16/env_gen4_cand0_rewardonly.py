@torch.jit.script
def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    object_linvel: torch.Tensor,
    progress_buf: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # --- Orientation error using quaternion dot product (shared strength) ---
    quat_diff = torch.abs(torch.sum(object_rot * goal_rot, dim=-1))
    quat_diff = torch.clamp(quat_diff, 0.0, 1.0)
    orientation_error = 1.0 - quat_diff

    orientation_temp = 10.0
    orientation_reward = torch.exp(-orientation_temp * orientation_error)

    # --- Linear velocity penalty to keep object stable (shared strength) ---
    linvel_penalty = torch.norm(object_linvel, dim=-1)
    linvel_temp = 2.0
    linvel_reward = torch.exp(-linvel_temp * linvel_penalty)

    # --- Enhanced angular velocity reward for better fine control ---
    angvel_norm = torch.norm(object_angvel, dim=-1)
    
    # Reduced proximity threshold to activate fine control earlier
    proximity_threshold = 0.05
    close_to_target = orientation_error <= proximity_threshold

    # Much stronger penalty when close to target to prevent wobbling/overshooting
    angvel_penalty_close = torch.exp(-20.0 * angvel_norm)

    # When far from target: still encourage moderate spinning but with slight penalty for excessive speed
    angvel_temp_far = 0.3
    angvel_reward_far = torch.exp(-angvel_temp_far * torch.abs(angvel_norm - 2.0))

    # Combine based on proximity to target
    angvel_reward = torch.where(close_to_target, angvel_penalty_close, angvel_reward_far)

    # --- Total reward: multiplicative for orientation and linear stability, additive scaled term for adaptive angular velocity ---
    total_reward = orientation_reward * linvel_reward + 0.2 * angvel_reward

    reward_components = {
        "orientation_reward": orientation_reward,
        "linvel_reward": linvel_reward,
        "angvel_reward": angvel_reward
    }

    return total_reward, reward_components
