@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the rotation error between object and goal orientation
    # quat_mul(a, b) gives a * b; to get relative rotation from object to goal: q_error = goal * inv(object)
    # Inverse of a unit quaternion is its conjugate
    quat_error = quat_mul(goal_rot, quat_conjugate(object_rot))
    
    # The scalar (w) component of the error quaternion relates to the angle difference
    # Rotation error in [0, 1], where 1 means perfect alignment
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_error[:, 1:], dim=1), min=0.0, max=1.0))  # angle in radians
    rot_reward = 1.0 - (rot_dist / torch.pi)  # normalized to [0, 1]

    # Optional: exponential shaping for sharper reward near target
    temp_rot = 1.0
    rot_reward_shaped = torch.exp(-temp_rot * rot_dist)

    reward = rot_reward_shaped

    reward_components = {
        "rot_reward": rot_reward,
        "rot_reward_shaped": rot_reward_shaped,
    }

    return reward, reward_components