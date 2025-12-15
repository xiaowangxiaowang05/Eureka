@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the rotation error between object and goal orientation
    # quat_mul(q, quat_conjugate(p)) gives the relative rotation from p to q
    rot_error = quat_mul(object_rot, quat_conjugate(goal_rot))
    
    # The angle of rotation error can be derived from the w component:
    # angle = 2 * acos(|w|), but we can use 1 - |w| as a proxy for small angles
    # Alternatively, use 1 - |w| or (1 - w^2) for smoothness; here we use 1 - abs(w)
    rot_dist = 1.0 - torch.abs(rot_error[:, 3])  # w component is at index 3

    # Temperature parameter for exponential transformation
    rot_temp = 1.0
    rot_reward = torch.exp(-rot_dist / rot_temp)

    reward = rot_reward
    reward_components = {
        "rot_reward": rot_reward
    }

    return reward, reward_components