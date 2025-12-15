def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    shadow_hand_dof_vel: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error via quaternion distance
    rel_rot = quat_mul(object_rot, quat_conjugate(goal_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(rel_rot[:, :3], p=2, dim=-1), max=1.0))
    
    # Dense orientation reward using exponential shaping
    rot_temp = 2.0
    rot_reward = torch.exp(-rot_temp * rot_dist)

    # Action regularization (smoothness/energy)
    action_norm = torch.sum(actions ** 2, dim=-1)
    action_penalty = -0.05 * action_norm  # small linear penalty instead of exp

    # Minor joint velocity penalty to reduce jitter (very light)
    joint_vel_norm = torch.sum(shadow_hand_dof_vel ** 2, dim=-1)
    joint_vel_penalty = -0.01 * joint_vel_norm

    # Total reward: dominant orientation term, minor penalties
    reward = rot_reward + action_penalty + joint_vel_penalty

    reward_components = {
        "rot_reward": rot_reward,
        "action_penalty": action_penalty,
        "joint_vel_penalty": joint_vel_penalty
    }

    return reward, reward_components


# Helper functions for TorchScript compatibility
@torch.jit.script
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([-q[:, :3], q[:, 3:4]], dim=-1)

@torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z2 * x1 - x2 * z1
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return torch.stack([x, y, z, w], dim=-1)