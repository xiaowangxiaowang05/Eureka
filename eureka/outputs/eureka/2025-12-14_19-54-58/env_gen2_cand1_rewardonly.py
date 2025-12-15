def compute_reward(
    object_rot: torch.Tensor,
    goal_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    progress_buf: torch.Tensor,
    max_episode_length: float
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute relative rotation quaternion: from goal to object
    rel_quat = quat_mul(object_rot, quat_conjugate(goal_rot))
    
    # Angular distance: ||log(q)|| = 2*arcsin(|imag(q)|)
    angle_error = 2.0 * torch.asin(torch.clamp(torch.norm(rel_quat[:, 0:3], dim=-1), min=0.0, max=1.0))
    
    # Dense orientation reward
    orientation_temp = 1.0
    orientation_reward = torch.exp(-orientation_temp * angle_error)
    
    # Spin maintenance around object Z-axis
    spin_axis = torch.zeros_like(object_angvel)
    spin_axis[:, 2] = 1.0
    spin_proj = torch.sum(object_angvel * spin_axis, dim=-1)
    spin_temp = 0.5
    spin_reward = torch.exp(-spin_temp * torch.abs(spin_proj - 5.0))
    
    # Action regularization
    action_penalty = -torch.sum(actions ** 2, dim=-1) * 0.01
    
    # Time-based timeout penalty
    time_factor = (progress_buf / max_episode_length).float()
    timeout_penalty = -time_factor * (1.0 - orientation_reward) * 0.1

    # Penalties for unnatural motion (assuming these are available in the environment class):
    # However, note: the function signature only includes the listed inputs.
    # Since joint acceleration, previous actions, and default pose are NOT in the input args,
    # we cannot directly penalize them here without modifying the environment.
    #
    # BUT WAIT: Looking at the environment code, the actual compute_reward() method is called with self.* attributes.
    # However, the problem states: "the reward function's input variables must contain only attributes of the provided environment class definition"
    # and the example signature uses only variables that appear in observation/state.
    #
    # In the PARENT CODE, the signature only included: object_rot, goal_rot, object_angvel, actions, progress_buf, max_episode_length
    # Therefore, we CANNOT add joint acceleration or action smoothness without those tensors being passed in.
    #
    # Re-examining the ENVIRONMENT CODE: the actual compute_reward(self, actions) method has access to:
    #   self.shadow_hand_dof_vel, self.prev_targets, self.shadow_hand_default_dof_pos, etc.
    # But the @torch.jit.script function must be standalone and only use its inputs.
    #
    # The instruction says: "the reward function's input variables must contain only attributes of the provided environment class definition (namely, variables that have prefix self.)"
    # AND the example shows input variables like `object_pos: torch.Tensor` which come from self.object_pos.
    #
    # However, the PARENT CODE signature did NOT include dof_vel or prev_actions, so we are constrained.
    #
    # Given the strict input signature from the parent, we must work within:
    #   object_rot, goal_rot, object_angvel, actions, progress_buf, max_episode_length
    #
    # Therefore, we cannot add acceleration penalty (needs prev_dof_vel) or action smoothness (needs prev_actions).
    # We also cannot add default pose deviation (needs shadow_hand_default_dof_pos and current dof_pos).
    #
    # BUT: the feedback says the policy exhibits high-frequency jitter and erratic motion.
    # One proxy we DO have is the action magnitude itself — high-frequency control often requires large, rapidly changing actions.
    # We already have an action penalty, but it may be too weak.
    #
    # Also, the spin_reward targets 5 rad/s, but maybe the agent is spinning too fast or in wrong axis.
    #
    # Let's adjust:
    # 1. Increase action penalty weight significantly to discourage excessive actuation.
    # 2. Add a penalty on angular velocity magnitude to prevent wild spinning.
    # 3. Make orientation reward more sensitive (lower temperature) to encourage precision.
    # 4. Remove the fixed target spin rate (5 rad/s) since the task is orientation, not sustained spinning.
    #
    # Revised plan:

    # Stronger action penalty to reduce jitter
    action_penalty = -torch.sum(actions ** 2, dim=-1) * 0.1  # increased from 0.01
    
    # Penalty on total angular speed to prevent uncontrolled spinning
    angvel_penalty = -torch.norm(object_angvel, dim=-1) * 0.05
    
    # More sensitive orientation reward
    orientation_temp = 2.0  # was 1.0
    orientation_reward = torch.exp(-orientation_temp * angle_error)
    
    # No specific target spin rate — just care about orientation
    spin_reward = torch.zeros_like(orientation_reward)  # disable spin_reward
    
    # Keep timeout penalty
    timeout_penalty = -time_factor * (1.0 - orientation_reward) * 0.1

    total_reward = (
        orientation_reward * 2.0 +
        spin_reward +
        action_penalty +
        angvel_penalty +
        timeout_penalty
    )
    
    reward_components = {
        "orientation_reward": orientation_reward,
        "spin_reward": spin_reward,
        "action_penalty": action_penalty,
        "angvel_penalty": angvel_penalty,
        "timeout_penalty": timeout_penalty
    }
    
    return total_reward, reward_components