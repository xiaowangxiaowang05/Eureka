def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    shadow_hand_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error using quaternion distance
    # quat_mul(q1, quat_conjugate(q2)) gives relative rotation from q2 to q1
    rel_quat = quat_mul(object_rot, quat_conjugate(goal_rot))
    # Quaternion error as angle (magnitude of rotation vector)
    # Extract w component; angle = 2 * acos(|w|), but we use 1 - |w| as a proxy for small angles
    rot_error = 1.0 - torch.abs(rel_quat[:, 3])  # w component is last in [x,y,z,w]?
    # Actually, in Isaac Gym, quaternions are usually [x, y, z, w], so index 3 is w
    rot_reward_temp = 2.0
    rot_reward = torch.exp(-rot_error / rot_reward_temp)

    # Encourage maintaining some angular velocity to keep spinning (avoid static grasp)
    # But don't over-spin; target a moderate angular speed
    angvel_norm = torch.norm(object_angvel, dim=-1)
    target_angvel = 2.0  # rad/s target spin rate
    angvel_error = torch.abs(angvel_norm - target_angvel)
    angvel_reward_temp = 1.0
    angvel_reward = torch.exp(-angvel_error / angvel_reward_temp)

    # Action regularization: penalize large actions
    action_penalty = torch.sum(actions ** 2, dim=-1)
    action_penalty_weight = 0.01
    action_reward = -action_penalty_weight * action_penalty

    # Joint velocity regularization: avoid jittery movements
    joint_vel_penalty = torch.sum(shadow_hand_dof_vel ** 2, dim=-1)
    joint_vel_weight = 0.001
    joint_vel_reward = -joint_vel_weight * joint_vel_penalty

    # Combine rewards
    total_reward = (
        rot_reward +
        0.5 * angvel_reward +
        action_reward +
        joint_vel_reward
    )

    reward_components = {
        "rot_reward": rot_reward,
        "angvel_reward": angvel_reward,
        "action_reward": action_reward,
        "joint_vel_reward": joint_vel_reward
    }

    return total_reward, reward_components


# Helper functions required for torch.jit.script compatibility
@torch.jit.script
def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[-1] == 4
    assert b.shape[-1] == 4

    x1, y1, z1, w1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    x2, y2, z2, w2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1)
    return quat

@torch.jit.script
def quat_conjugate(a: torch.Tensor) -> torch.Tensor:
    conj = torch.cat([-a[..., :3], a[..., 3:]], dim=-1)
    return conj