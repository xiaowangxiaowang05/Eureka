def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    shadow_hand_dof_vel: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute rotational distance using quaternion difference norm (more stable and dense)
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

    # Dense exponential reward for orientation alignment
    rot_temp = 1.0
    rot_reward = torch.exp(-rot_temp * rot_dist)

    # Encourage spinning with appropriate angular velocity magnitude
    angvel_norm = torch.norm(object_angvel, dim=-1)
    target_angvel = 2.0
    angvel_error = torch.abs(angvel_norm - target_angvel)
    angvel_temp = 0.5
    angvel_reward = torch.exp(-angvel_temp * angvel_error)

    # Very light action regularization to avoid unnecessary movements
    action_penalty = torch.sum(actions ** 2, dim=-1)
    action_reg_weight = 0.0001  # Reduced from 0.001
    action_reward = -action_reg_weight * action_penalty

    # Remove joint velocity penalty entirely to avoid freezing behavior
    joint_vel_reward = torch.zeros_like(action_reward)

    total_reward = rot_reward + angvel_reward + action_reward + joint_vel_reward

    reward_components = {
        "rot_reward": rot_reward,
        "angvel_reward": angvel_reward,
        "action_reward": action_reward,
        "joint_vel_reward": joint_vel_reward
    }

    return total_reward, reward_components

# Helper functions required by TorchScript
@torch.jit.script
def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[-1] == 4
    assert b.shape[-1] == 4

    x1, y1, z1, w1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    x2, y2, z2, w2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([x, y, z, w], dim=-1)

@torch.jit.script
def quat_conjugate(a: torch.Tensor) -> torch.Tensor:
    conj = torch.cat([-a[..., 0:3], a[..., 3:4]], dim=-1)
    return conj