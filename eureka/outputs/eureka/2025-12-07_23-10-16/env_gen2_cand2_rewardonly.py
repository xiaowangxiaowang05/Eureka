@torch.jit.script
def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    object_linvel: torch.Tensor,
    progress_buf: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # --- Orientation error using quaternion dot product (from Code A) ---
    quat_diff = torch.abs(torch.sum(object_rot * goal_rot, dim=-1))
    quat_diff = torch.clamp(quat_diff, 0.0, 1.0)
    orientation_error = 1.0 - quat_diff

    orientation_temp = 10.0
    orientation_reward = torch.exp(-orientation_temp * orientation_error)

    # --- Linear velocity penalty to keep object stable (from Code A) ---
    linvel_penalty = torch.norm(object_linvel, dim=-1)
    linvel_temp = 2.0
    linvel_reward = torch.exp(-linvel_temp * linvel_penalty)

    # --- Adaptive angular velocity reward (inspired by Code B) ---
    angvel_norm = torch.norm(object_angvel, dim=-1)
    proximity_threshold = 0.1
    close_to_target = orientation_error <= proximity_threshold

    # When close: penalize any motion strongly
    angvel_penalty_close = torch.exp(-5.0 * angvel_norm)

    # When far: encourage moderate spinning (like Code A but with better shaping)
    angvel_reward_far = torch.exp(-0.5 * torch.abs(angvel_norm - 2.0))

    # Combine based on proximity to target
    angvel_reward = torch.where(close_to_target, angvel_penalty_close, angvel_reward_far)

    # --- Total reward: multiplicative for stability + orientation, additive for adaptive angvel ---
    total_reward = orientation_reward * linvel_reward + 0.2 * angvel_reward

    reward_components = {
        "orientation_reward": orientation_reward,
        "linvel_reward": linvel_reward,
        "angvel_reward": angvel_reward
    }

    return total_reward, reward_components
