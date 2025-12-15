def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    shadow_hand_dof_vel: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error using quaternion distance
    # quat_mul(q1, quat_conjugate(q2)) gives relative rotation from q2 to q1
    rel_quat = torch.mul(object_rot, quat_conjugate(goal_rot))
    # The scalar part (w component) of the relative quaternion indicates alignment
    # When object_rot == goal_rot, rel_quat = [0,0,0,1], so w=1
    rot_dist = 1.0 - rel_quat[:, 3]  # ranges from 0 (aligned) to 2 (opposite)
    
    # Temperature parameter for orientation reward shaping
    rot_temp: float = 1.0
    rot_reward = torch.exp(-rot_temp * rot_dist)

    # Encourage appropriate angular velocity during spinning
    # For spinning tasks, some angular velocity is needed but excessive velocity may be unstable
    angvel_norm = torch.norm(object_angvel, dim=-1)
    target_angvel: float = 2.0  # moderate target spin speed
    angvel_error = torch.abs(angvel_norm - target_angvel)
    angvel_temp: float = 0.5
    angvel_reward = torch.exp(-angvel_temp * angvel_error)

    # Action regularization to reduce jitter and energy use
    action_penalty = torch.sum(actions ** 2, dim=-1)
    action_temp: float = 0.01
    action_reward = -action_temp * action_penalty

    # Joint velocity regularization to prevent extreme movements
    joint_vel_norm = torch.sum(shadow_hand_dof_vel ** 2, dim=-1)
    joint_vel_temp: float = 0.001
    joint_vel_reward = -joint_vel_temp * joint_vel_norm

    # Combine rewards with weights
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


# Helper functions needed for TorchScript compatibility
@torch.jit.script
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    conj = torch.cat([-q[:, :3], q[:, 3:4]], dim=-1)
    return conj

@torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([x, y, z, w], dim=-1)