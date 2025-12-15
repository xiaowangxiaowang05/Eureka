@torch.jit.script
def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    shadow_hand_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute rotation error via quaternion inner product (more stable than norm difference)
    quat_inner = torch.sum(object_rot * goal_rot, dim=-1)
    rot_error = 1.0 - torch.abs(quat_inner)  # [0, 1]

    # Dense orientation reward with temperature scaling
    rot_reward_temp = 3.0
    rot_reward = torch.exp(-rot_reward_temp * rot_error)

    # Encourage non-zero angular velocity to promote spinning (not just static alignment)
    angvel_norm = torch.norm(object_angvel, dim=-1)
    min_angvel = 1.5  # minimum angular speed to be considered "spinning"
    angvel_reward = torch.clamp(angvel_norm - min_angvel, min=0.0)  # linear bonus above threshold

    # Light action regularization to avoid jitter, but not so strong it freezes motion
    action_penalty = torch.sum(actions ** 2, dim=-1)
    action_penalty_weight = 0.001
    action_reward = -action_penalty_weight * action_penalty

    # Remove joint velocity penalty entirely to avoid freezing behavior

    total_reward = (
        rot_reward +
        0.5 * angvel_reward +
        action_reward
    )

    reward_components = {
        "rot_reward": rot_reward,
        "angvel_reward": angvel_reward,
        "action_reward": action_reward
    }

    return total_reward, reward_components