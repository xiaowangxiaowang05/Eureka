@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, object_angvel: torch.Tensor, progress_buf: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error using quaternion distance
    # The rotation error can be computed as the angle of the relative rotation
    # quat_mul(object_rot, quat_conjugate(goal_rot)) gives the error quaternion
    # The scalar part (w) of this quaternion relates to the rotation angle
    
    # Compute error quaternion: q_error = object_rot * conjugate(goal_rot)
    # For quaternions [x, y, z, w], conjugate is [-x, -y, -z, w]
    goal_rot_conj = torch.cat([-goal_rot[:, :3], goal_rot[:, 3:4]], dim=-1)
    
    # Quaternion multiplication: q1 * q2
    # q_result.w = q1.w*q2.w - q1.x*q2.x - q1.y*q2.y - q1.z*q2.z
    # q_result.xyz = q1.w*q2.xyz + q2.w*q1.xyz + cross(q1.xyz, q2.xyz)
    w1 = object_rot[:, 3:4]
    xyz1 = object_rot[:, :3]
    w2 = goal_rot_conj[:, 3:4]
    xyz2 = goal_rot_conj[:, :3]
    
    w_error = w1 * w2 - torch.sum(xyz1 * xyz2, dim=-1, keepdim=True)
    xyz_error = w1 * xyz2 + w2 * xyz1 + torch.cross(xyz1, xyz2, dim=-1)
    
    # The rotation angle theta satisfies: cos(theta/2) = |w_error|
    # So the angle error is: theta = 2 * acos(|w_error|)
    # To avoid numerical issues, clamp w_error to [-1, 1]
    w_error_clamped = torch.clamp(torch.abs(w_error), min=0.0, max=1.0)
    rot_angle_error = 2.0 * torch.acos(w_error_clamped)
    
    # Alternative simpler approach: use 1 - |dot product| as orientation error
    # This is more stable and commonly used in RL
    rot_error = 1.0 - torch.abs(torch.sum(object_rot * goal_rot, dim=-1, keepdim=True))
    
    # Angular velocity penalty to prevent excessive spinning
    # We want controlled spinning, not wild rotation
    angvel_penalty = torch.norm(object_angvel, dim=-1, keepdim=True) * 0.05
    
    # Temperature parameters for reward shaping
    rot_temp = 1.0
    angvel_temp = 0.1
    
    # Compute individual reward components
    rot_reward = torch.exp(-rot_error / rot_temp)
    angvel_reward = torch.exp(-angvel_penalty / angvel_temp)
    
    # Total reward combines orientation accuracy and controlled motion
    total_reward = rot_reward * angvel_reward
    
    reward_components = {
        "rot_reward": rot_reward.squeeze(-1),
        "angvel_reward": angvel_reward.squeeze(-1),
        "rot_error": rot_error.squeeze(-1)
    }
    
    return total_reward.squeeze(-1), reward_components
