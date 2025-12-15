@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the rotation error between object and goal orientation
    # quat_mul(q1, quat_conjugate(q2)) gives the relative rotation from q2 to q1
    rot_error = quat_mul(object_rot, quat_conjugate(goal_rot))
    
    # The angle of rotation error can be derived from the w component:
    # angle = 2 * acos(|w|), but we can use 1 - |w| as a smooth approximation of alignment
    rot_dist = 1.0 - torch.abs(rot_error[:, 3])  # w component is at index 3

    # Temperature parameter for exponential reward shaping
    rot_reward_temp = 1.0
    rot_reward = torch.exp(-rot_dist / rot_reward_temp)

    reward = rot_reward

    reward_components = {
        "rot_reward": rot_reward
    }

    return reward, reward_components