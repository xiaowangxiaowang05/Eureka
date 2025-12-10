@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, object_angvel: torch.Tensor, progress_buf: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute rotation error using quaternion distance
    # quat_mul(object_rot, quat_conjugate(goal_rot)) gives the relative rotation
    # The scalar part (w component) of this relative rotation indicates alignment
    # When object_rot == goal_rot, the relative rotation is [0, 0, 0, 1] (identity)
    
    # Compute conjugate of goal rotation
    goal_rot_conj = torch.cat([-goal_rot[:, :3], goal_rot[:, 3:4]], dim=-1)
    
    # Compute relative rotation: object_rot * goal_rot_conj
    w_rel = (object_rot[:, 3] * goal_rot_conj[:, 3] - 
             object_rot[:, 0] * goal_rot_conj[:, 0] - 
             object_rot[:, 1] * goal_rot_conj[:, 1] - 
             object_rot[:, 2] * goal_rot_conj[:, 2])
    
    # Rotation alignment reward (closer to 1 means better alignment)
    rot_error = 1.0 - torch.abs(w_rel)
    
    # Temperature parameter for rotation reward
    rot_temp = 1.0
    
    # Exponential reward for rotation alignment
    rot_reward = torch.exp(-rot_error / rot_temp)
    
    # Small bonus for maintaining angular velocity (encourages spinning motion)
    angvel_norm = torch.norm(object_angvel, dim=-1)
    angvel_temp = 0.1
    angvel_reward = torch.exp(-angvel_norm / angvel_temp) * 0.1
    
    # Total reward
    total_reward = rot_reward + angvel_reward
    
    reward_components = {
        "rot_reward": rot_reward,
        "angvel_reward": angvel_reward
    }
    
    return total_reward, reward_components