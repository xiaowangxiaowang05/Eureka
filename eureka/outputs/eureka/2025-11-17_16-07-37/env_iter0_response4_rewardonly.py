@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dof_vel: torch.Tensor, actions: torch.Tensor, up_vec: torch.Tensor, contact_force_tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract relevant state information
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    
    # Compute forward direction (assuming y-axis is forward in the environment's coordinate system)
    # Based on observation code, the target direction is projected onto x-y plane (z=0)
    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    target_dist = torch.norm(to_target, p=2, dim=-1)
    
    # Forward velocity reward - encourage moving in the direction of the target
    target_dir = to_target / (target_dist.unsqueeze(-1) + 1e-8)
    forward_vel = torch.sum(velocity * target_dir, dim=-1)
    
    # Velocity reward with temperature scaling
    vel_reward_temp = 0.1
    vel_reward = torch.exp(forward_vel * vel_reward_temp)
    
    # Upright posture reward - penalize deviation from upright orientation
    # up_vec is computed in compute_humanoid_observations and represents the up vector projection
    upright_reward_temp = 1.0
    upright_reward = torch.exp(up_vec[:, 2] * upright_reward_temp)
    
    # Action rate penalty - penalize large changes in actions (smoothness)
    action_rate_penalty_temp = 0.1
    action_rate_penalty = torch.sum(torch.square(actions), dim=-1)
    action_rate_reward = torch.exp(-action_rate_penalty * action_rate_penalty_temp)
    
    # Joint velocity penalty - discourage excessive joint velocities
    joint_vel_penalty_temp = 0.01
    joint_vel_penalty = torch.sum(torch.square(dof_vel), dim=-1)
    joint_vel_reward = torch.exp(-joint_vel_penalty * joint_vel_penalty_temp)
    
    # Contact force penalty - discourage excessive contact forces (stability)
    contact_force_penalty_temp = 0.001
    contact_force_penalty = torch.sum(torch.square(contact_force_tensor), dim=-1)
    contact_force_reward = torch.exp(-contact_force_penalty * contact_force_penalty_temp)
    
    # Combine rewards with weights
    total_reward = (
        2.0 * vel_reward +
        1.0 * upright_reward +
        0.5 * action_rate_reward +
        0.5 * joint_vel_reward +
        0.5 * contact_force_reward
    )
    
    reward_dict = {
        "vel_reward": vel_reward,
        "upright_reward": upright_reward,
        "action_rate_reward": action_rate_reward,
        "joint_vel_reward": joint_vel_reward,
        "contact_force_reward": contact_force_reward
    }
    
    return total_reward, reward_dict
