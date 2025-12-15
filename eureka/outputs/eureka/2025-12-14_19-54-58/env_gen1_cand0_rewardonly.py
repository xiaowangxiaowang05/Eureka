@torch.jit.script
def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    shadow_hand_dof_vel: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # This function cannot work as intended because it lacks access to critical observations:
    # - fingertip positions (to reward grasp)
    # - object position (to prevent dropping)
    # However, per instructions we must only use inputs from parent code.
    # Given this constraint, we reinterpret "object_angvel" as a proxy for engagement 
    # and strengthen orientation shaping while reducing penalties that discourage motion.
    
    # Compute orientation alignment reward using quaternion dot product
    rel_quat = quat_mul(object_rot, quat_conjugate(goal旋_goal))
    rot_error = 1.0 - torch.abs(rel_quat[:, 3])  # 0=perfect, 1=worst
    
    orientation_temp = 0.5
    orientation_reward = torch.exp(-orientation_temp * rot_error * 10.0)
    
    # Encourage rotation initiation: reward angular velocity magnitude
    angvel_norm = torch.norm(object_angvel, dim=-1)
    angvel_reward = torch.tanh(angvel_norm * 2.0)  # Saturation at ~0.5 rad/s
    
    # Action regularization - significantly reduced to avoid freezing
    action_penalty = torch.sum(actions ** 2, dim=-1)
    action_reward = -0.001 * action_penalty
    
    # Joint velocity penalty - very small to allow necessary motion
    joint_vel_norm = torch.sum(shadow_hand_dof_vel ** 2, dim=-1)
    joint_vel_reward = -0.0001 * joint_vel_norm
    
    total_reward = (
        orientation_reward * 2.0 +
        angvel_reward * 1.0 +
        action_reward +
        joint_vel_reward
    )
    
    return total_reward, {
        "orientation_reward": orientation_reward,
        "angvel_reward": angvel_reward,
        "action_reward": action_reward,
        "joint_vel_reward": joint_vel_reward
    }