def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    shadow_hand_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute rotation error using angle-axis distance on SO(3)
    rel_quat = quat_mul(object_rot, quat_conjugate(goal_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(rel_quat[:, :3], dim=-1), max=1.0))
    
    # Dense orientation reward: higher when closer to target orientation
    rot_reward_scale = 2.0
    rot_reward = torch.exp(-rot_dist * rot_reward_scale)
    
    # Encourage appropriate spinning: align angular velocity with desired rotation axis
    # Desired axis is extracted from relative quaternion
    axis_angle = quat_to_angle_axis(rel_quat)
    desired_axis = axis_angle[1]  # shape: (num_envs, 3)
    desired_speed = axis_angle[0]  # shape: (num_envs,)
    
    # Project actual angvel onto desired axis
    angvel_on_axis = torch.sum(object_angvel * desired_axis, dim=-1)
    # Reward alignment and moderate speed (avoid excessive spin)
    angvel_target = torch.clamp(desired_speed / 0.1, min=0.0, max=5.0)  # reasonable target speed
    angvel_error = torch.abs(angvel_on_axis - angvel_target)
    angvel_reward = torch.exp(-angvel_error * 0.5)
    
    # Action regularization (quadratic penalty, not exponential)
    action_penalty = -0.01 * torch.sum(actions ** 2, dim=-1)
    
    # Joint velocity penalty to avoid jitter (modest)
    joint_vel_penalty = -0.005 * torch.sum(shadow_hand_dof_vel ** 2, dim=-1)
    
    # Combine components
    total_reward = (
        rot_reward +
        0.5 * angvel_reward +
        action_penalty +
        joint_vel_penalty
    )
    
    reward_components = {
        "rot_reward": rot_reward,
        "angvel_reward": angvel_reward,
        "action_penalty": action_penalty,
        "joint_vel_penalty": joint_vel_penalty
    }
    
    return total_reward, reward_components

# Required helper functions for TorchScript
@torch.jit.script
def quat_conjugate(q):
    return torch.cat((-q[:, :3], q[:, 3:4]), dim=-1)

@torch.jit.script
def quat_mul(q1, q2):
    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([x, y, z, w], dim=-1)

@torch.jit.script
def quat_to_angle_axis(q):
    # q: [w, x, y, z] convention in some contexts, but our q is [x,y,z,w]
    # So extract w = q[:,3], xyz = q[:,:3]
    angle = 2.0 * torch.acos(torch.clamp(q[:, 3], min=-1.0, max=1.0))
    sin_half_angle = torch.sqrt(1.0 - q[:, 3] * q[:, 3])
    axis = torch.where(
        sin_half_angle.unsqueeze(-1) > 1e-6,
        q[:, :3] / sin_half_angle.unsqueeze(-1),
        torch.tensor([1.0, 0.0, 0.0], device=q.device).repeat(q.shape[0], 1)
    )
    return angle, axis