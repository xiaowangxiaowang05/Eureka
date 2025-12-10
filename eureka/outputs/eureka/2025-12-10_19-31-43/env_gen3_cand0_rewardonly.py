def compute_reward(
    object_pos: torch.Tensor,
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    progress_buf: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Keep object close to hand center (approximate hand position at origin + offset)
    # In this environment, the hand starts around (0, 0, 0.5), object starts at (0, -0.39, 0.6)
    # But for simplicity and generalization, we encourage object to stay near initial grasp region
    # Use distance from a reference point (e.g., [0, -0.4, 0.6]) or just relative stability
    # However, since no explicit hand pos is passed, we use object position stability as proxy
    # But more importantly, prevent object from falling too far (z < threshold)
    # Instead, use distance from initial expected position as implicit "stay in hand" signal
    ref_pos = torch.tensor([0.0, -0.4, 0.6], device=object_pos.device).unsqueeze(0)
    pos_dist = torch.norm(object_pos - ref_pos, dim=1)
    pos_temp = 2.0
    pos_reward = torch.exp(-pos_temp * pos_dist)

    # Rotational alignment reward
    rot_error_quat = quat_mul(object_rot, quat_conjugate(goal_rot))
    rot_error_w = rot_error_quat[:, 3]
    rot_error_w = torch.clamp(rot_error_w, -1.0, 1.0)
    rot_angle_error = 2.0 * torch.acos(torch.abs(rot_error_w))
    rot_temp = 1.0
    rot_reward = torch.exp(-rot_temp * rot_angle_error)

    # Angular velocity reward: encourage spinning but not too fast
    angvel_norm = torch.norm(object_angvel, dim=1)
    optimal_angvel = 2.0
    angvel_error = torch.abs(angvel_norm - optimal_angvel)
    angvel_temp = 0.1
    angvel_reward = torch.exp(-angvel_temp * angvel_error)

    # Action penalty to discourage excessive inactivity OR thrashing
    # But also ensure some action is taken (avoid zero action freeze)
    action_norm = torch.norm(actions, dim=1)
    action_target = 0.5  # encourage moderate action magnitude
    action_error = torch.abs(action_norm - action_target)
    action_temp = 0.5
    action_reward = torch.exp(-action_temp * action_error)

    # Time penalty to encourage efficiency
    time_penalty = -0.01 * progress_buf.float()

    # Combine rewards
    total_reward = (
        1.0 * pos_reward +
        2.0 * rot_reward +
        0.3 * angvel_reward +
        0.5 * action_reward +
        time_penalty
    )

    reward_components = {
        "pos_reward": pos_reward,
        "rot_reward": rot_reward,
        "angvel_reward": angvel_reward,
        "action_reward": action_reward,
        "time_penalty": time_penalty
    }

    return total_reward, reward_components

# Helper functions (must be included for @torch.jit.script compatibility)
@torch.jit.script
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([-q[:, :3], q[:, 3:4]], dim=1)

@torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return torch.stack([x, y, z, w], dim=1)