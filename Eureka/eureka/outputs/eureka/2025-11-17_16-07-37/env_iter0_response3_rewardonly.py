@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dof_vel: torch.Tensor, actions: torch.Tensor, contact_force_tensor: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract linear velocity in world frame
    velocity = root_states[:, 7:10]
    
    # Compute forward direction (y-axis in this humanoid setup based on basis_vec1)
    # Assuming the humanoid moves along the y-axis as forward direction
    forward_vel = velocity[:, 1]  # y-component of linear velocity
    
    # Main goal reward: encourage high forward speed
    run_reward = forward_vel
    
    # Penalty for excessive action magnitude (encourage energy efficiency)
    action_cost = -torch.sum(actions ** 2, dim=-1) * 0.01
    
    # Penalty for high joint velocities (smoothness)
    joint_vel_penalty = -torch.sum(dof_vel ** 2, dim=-1) * 0.0001
    
    # Penalty for contact forces (discourage harsh impacts)
    contact_penalty = -torch.sum(contact_force_tensor.view(-1, 24) ** 2, dim=-1) * 0.00001
    
    # Total reward
    total_reward = run_reward + action_cost + joint_vel_penalty + contact_penalty
    
    reward_dict = {
        "run_reward": run_reward,
        "action_cost": action_cost,
        "joint_vel_penalty": joint_vel_penalty,
        "contact_penalty": contact_penalty
    }
    
    return total_reward, reward_dict
