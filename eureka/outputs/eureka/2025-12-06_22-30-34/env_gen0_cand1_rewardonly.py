@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, object_angvel: torch.Tensor, progress_buf: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error using quaternion distance
    # The rotation error can be computed as the angle of the relative rotation
    # quat_mul(object_rot, quat_conjugate(goal_rot)) gives the error quaternion
    # The scalar part (w) of this quaternion relates to the rotation angle
    
    # Compute conjugate of goal_rot
    goal_rot_conj = torch.cat([-goal_rot[:, 1:4], goal_rot[:, 0:1]], dim=1)
    
    # Compute error quaternion: object_rot * goal_rot_conj
    w1, x1, y1, z1 = object_rot[:, 0], object_rot[:, 1], object_rot[:, 2], object_rot[:, 3]
    w2, x2, y2, z2 = goal_rot_conj[:, 0], goal_rot_conj[:, 1], goal_rot_conj[:, 2], goal_rot_conj[:, 3]
    
    err_w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    # err_x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    # err_y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    # err_z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    # The absolute value of the scalar part gives cos(theta/2)
    # So the orientation error is proportional to (1 - |err_w|)
    orientation_error = 1.0 - torch.abs(err_w)
    
    # Reward for orientation alignment (higher when error is smaller)
    orientation_reward_temp = 0.1
    orientation_reward = torch.exp(-orientation_error / orientation_reward_temp)
    
    # Encourage some spinning motion (angular velocity magnitude)
    angvel_magnitude = torch.norm(object_angvel, dim=1)
    spin_reward_temp = 1.0
    spin_reward = torch.exp(-torch.abs(angvel_magnitude - 2.0) / spin_reward_temp)  # Target moderate spin
    
    # Combine rewards
    total_reward = orientation_reward + 0.1 * spin_reward
    
    reward_components = {
        "orientation_reward": orientation_reward,
        "spin_reward": spin_reward
    }
    
    return total_reward, reward_components
