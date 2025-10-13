from mujoco_mpc import agent as agent_lib
import numpy as np

class MPCPlanner():
    """
    A simple MPC planner using mujoco_mpc's Agent for trajectory optimization.

    TODO: Template up to change!

    TODO: Model this after the mjpc_ex.py example script to get a feeling for things
    """

    def __init__(self, model, task_id="Cartpole", horizon=20):
        self.agent = agent_lib.Agent(task_id=task_id, model=model)
        self.horizon = horizon
    
    def plan(self, current_state):
        """
        Plan an action sequence from the current state using MPC.
        Args:
            current_state: The current observation/state of the environment.
        Returns:
            action: The first action in the optimized action sequence.
        """
        # Set the agent's state to the current state
        self.agent.set_state(current_state)
        
        # Optimize the trajectory
        self.agent.optimize(self.horizon)
        
        # Get the optimized action sequence
        action_sequence = self.agent.get_action_sequence()
        
        # Return the first action
        return action_sequence[0]