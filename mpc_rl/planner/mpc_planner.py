from mujoco_mpc import agent as agent_lib
import numpy as np
import mujoco
import pathlib
from gymnasium_robotics.utils.rotations import quat_mul

class MPCPlanner():
    """
    A simple MPC planner using mujoco_mpc's Agent for trajectory optimization.

    Default configuration is for the cartpole swingup task.
    """

    def __init__(self,
                 model_path=None,
                 task_id="Cartpole",
                 rollout_horizon=10000,
                 opt_steps=10,
                 weights: dict[str, float] = {},
                 task_params: dict[str, float] = {},
                 init_state_noise_flag=False,
                 qpos_noise_rnge=(),
                 qvel_noise_rnge=(),
                 verbose: int=0
        ) -> None:
        """
        model_path: Path to the task XML model file.
        task_id: Identifier for the task (e.g., "Cartpole").
        rollout_horizon: Total length of the trajectory/episode.
        opt_steps: Number of optimization steps for the MPC agent.
        
        NOTE: The planner parameters, e.g. planner type, agent horizon, agent timestep, are set in each task.xml file.

        Agent horizon is the prediction horizon the MPC uses while the rollout horizon here is the length of an entire trajectory/episode.
        """
        if model_path is None:
            self.model_path = (
                pathlib.Path(__file__).parent.parent.parent
                / "mpc_rl/tasks/cartpole/task.xml"
            )
        else:
            self.model_path = model_path

        self.verbose = verbose
        self.task_id = task_id
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        # NOTE:
        # agent class has exposed functions to:
        # - Set cost weights
        # - Set task parameters (e.g. goal position)
        self.agent = agent_lib.Agent(task_id=self.task_id, model=self.model)
        
        # Set cost weights and task parameters for any task
        if weights:
            self.agent.set_cost_weights(weights)
        if task_params:
            self.agent.set_task_parameters(task_params)
        
        self.rollout_horizon = rollout_horizon
        self.opt_steps = opt_steps
        self.init_state_noise_flag = init_state_noise_flag
        self.qpos_noise_rnge = qpos_noise_rnge
        self.qvel_noise_rnge = qvel_noise_rnge

        # Calculate agent timestep ratio (how many physics steps per agent update)
        # This is needed to properly record controls at agent timestep intervals
        self.physics_timestep = self.model.opt.timestep
        self.agent_timestep = self._get_agent_timestep_from_model()
        self.steps_per_agent_update = int(round(self.agent_timestep / self.physics_timestep))
        
        # Trajectories
        self.qpos = np.zeros((self.model.nq, self.rollout_horizon))
        self.qvel = np.zeros((self.model.nv, self.rollout_horizon))
        self.ctrl = np.zeros((self.model.nu, self.rollout_horizon - 1))
        self.time = np.zeros(self.rollout_horizon)

        # Costs
        self.cost_total = np.zeros(self.rollout_horizon - 1)
        self.cost_terms = np.zeros((len(self.agent.get_cost_term_values()), self.rollout_horizon - 1))

        # Reset data (for later rollout just in case)
        mujoco.mj_resetData(self.model, self.data)
    
    def _get_agent_timestep_from_model(self) -> float:
        """Extract agent_timestep from the model's custom numeric data."""
        # Find the agent_timestep custom numeric in the model
        for i in range(self.model.nnumeric):
            numeric_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_NUMERIC, i)
            if numeric_name == "agent_timestep":
                return self.model.numeric_data[self.model.numeric_adr[i]]
        # Default fallback if not found
        return self.physics_timestep
    
    def _randomize_shadow_cube_and_goal(self):
        """
        Randomize Shadow Reorient task cube and goal poses.
        This follows the manipulate_block.py specification from gymnasium_robotics.
        
        NOTE: Random seed should be set before calling this method.
        """
        # Randomize manipulated cube's initial pose
        cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        cube_jnt_addr = self.model.body_jntadr[cube_body_id]
        cube_qpos_start = self.model.jnt_qposadr[cube_jnt_addr]
        
        # Randomize position: base position + Gaussian noise (mean=0, std=0.005)
        base_pos = np.array([0.325, 0.0, 0.075])
        pos_noise = np.random.normal(0, 0.005, size=3)
        random_pos = base_pos + pos_noise
        
        # Randomize orientation: random rotation around random axis
        random_angle = np.random.uniform(-np.pi, np.pi)
        axis_idx = np.random.randint(0, 3)
        axis = np.zeros(3)
        axis[axis_idx] = 1.0
        
        # Convert axis-angle to quaternion
        half_angle = random_angle / 2
        sin_half = np.sin(half_angle)
        cube_init_quat = np.array([
            np.cos(half_angle),       # w
            axis[0] * sin_half,       # x
            axis[1] * sin_half,       # y
            axis[2] * sin_half        # z
        ])
        
        # Set the randomized pose in qpos
        self.data.qpos[cube_qpos_start:cube_qpos_start+3] = random_pos
        self.data.qpos[cube_qpos_start+3:cube_qpos_start+7] = cube_init_quat
        
        # Randomize goal cube's orientation
        goal_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal")
        goal_jnt_addr = self.model.body_jntadr[goal_body_id]
        goal_qpos_start = self.model.jnt_qposadr[goal_jnt_addr]
        
        # Sample orientation offset: random angle on a random axis
        goal_angle_offset = np.random.uniform(-np.pi, np.pi)
        goal_axis_idx = np.random.randint(0, 3)
        goal_axis = np.zeros(3)
        goal_axis[goal_axis_idx] = 1.0
        
        # Convert offset axis-angle to quaternion
        half_offset = goal_angle_offset / 2
        sin_half_offset = np.sin(half_offset)
        offset_quat = np.array([
            np.cos(half_offset),
            goal_axis[0] * sin_half_offset,
            goal_axis[1] * sin_half_offset,
            goal_axis[2] * sin_half_offset
        ])
        
        # Multiply quaternions: goal_quat = offset_quat * cube_init_quat
        goal_quat = quat_mul(offset_quat, cube_init_quat)
        self.data.qpos[goal_qpos_start:goal_qpos_start+4] = goal_quat
        
        # Forward kinematics to update both cubes
        mujoco.mj_forward(self.model, self.data)
        
        if self.verbose > 0:
            print(f"Randomized cube initial pose:")
            print(f"  Position: {random_pos}")
            print(f"  Quaternion: {cube_init_quat}")
            print(f"  Rotation axis: {['X', 'Y', 'Z'][axis_idx]}, angle: {random_angle:.3f} rad ({np.degrees(random_angle):.1f}°)")
            print(f"Goal cube orientation:")
            print(f"  Quaternion: {goal_quat}")
            print(f"  Offset axis: {['X', 'Y', 'Z'][goal_axis_idx]}, offset angle: {goal_angle_offset:.3f} rad ({np.degrees(goal_angle_offset):.1f}°)")
    
    def plan(self, keyframe: str="home", init_qpos=None, init_qvel=None, seed=None) -> None:
        """
        Plan an action sequence from the keyframe using MPC.
        NOTE: The keyframe is defined in the task XML file. The "home" keyframe is usually the default starting state. So if you want to start from a different initial state, define a new keyframe in the XML for now.
        
        The ctrl array will be at physics timestep resolution, but controls only update
        at agent_timestep intervals (with the same action held constant between updates).
        
        Args:
            keyframe: Name of the keyframe to start from
            init_qpos: Optional initial joint positions (overrides keyframe)
            init_qvel: Optional initial joint velocities (overrides keyframe)
            seed: Optional random seed for state noise initialization (used when init_state_noise_flag=True)
        """
        # Reset data
        mujoco.mj_resetData(self.model, self.data)

        # Reset to specified keyframe
        keyframe_id = mujoco.mj_name2id(self.model,
                                        mujoco.mjtObj.mjOBJ_KEY,
                                        keyframe)
        if keyframe_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, keyframe_id)
        else:
            print(f"Warning: '{keyframe}' keyframe not found in model")
        
        # Apply initialization noise if requested
        if self.init_state_noise_flag:
            # Set random seed for reproducibility
            if seed is not None:
                np.random.seed(seed)
            
            # For Shadow task, use custom randomization
            if self.task_id == "Shadow":
                self._randomize_shadow_cube_and_goal()
            else:
                # For other tasks, use standard qpos/qvel noise
                self.qpos_noise = np.random.uniform(self.qpos_noise_rnge[0],
                                                    self.qpos_noise_rnge[1],
                                                    size=self.model.nq)
                self.qvel_noise = np.random.uniform(self.qvel_noise_rnge[0],
                                                    self.qvel_noise_rnge[1],
                                                    size=self.model.nv)
                self.data.qpos[:] += self.qpos_noise
                self.data.qvel[:] += self.qvel_noise

        # If initial qpos/qvel are provided, override the keyframe state
        if init_qpos is not None:
            self.data.qpos[:] = init_qpos
        if init_qvel is not None:
            self.data.qvel[:] = init_qvel

        # Cache initial state
        self.qpos[:, 0] = self.data.qpos
        self.qvel[:, 0] = self.data.qvel
        self.time[0] = self.data.time

        # Simulate over the rollout horizon
        for t in range(self.rollout_horizon - 1):
            if self.verbose > 0 and t % 100 == 0:
                print(f"Planning step {t}/{self.rollout_horizon}")

            # Only update agent planner at agent_timestep intervals
            if t % self.steps_per_agent_update == 0:
                # Set planner state
                self.agent.set_state(
                    time=self.data.time,
                    qpos=self.data.qpos,
                    qvel=self.data.qvel,
                    act=self.data.act,
                    mocap_pos=self.data.mocap_pos,
                    mocap_quat=self.data.mocap_quat,
                    userdata=self.data.userdata
                )

                # Run planner optimization step
                for _ in range(self.opt_steps):
                    self.agent.planner_step()

                # Get new action from agent
                self.data.ctrl = self.agent.get_action()

            # If not updating agent, ctrl remains the same (held from previous update)
            # Record control at every physics step
            self.ctrl[:, t] = self.data.ctrl

            # Get costs
            self.cost_total[t] = self.agent.get_total_cost()
            for i, c in enumerate(self.agent.get_cost_term_values().items()):
                self.cost_terms[i, t] = c[1]
            
            # Step the simulation
            mujoco.mj_step(self.model, self.data)

            # Cache the states and time
            self.qpos[:, t+1] = self.data.qpos
            self.qvel[:, t+1] = self.data.qvel
            self.time[t+1] = self.data.time

        # Reset the agent
        self.agent.reset()

    def plan_receding_horizon(self, keyframe: str="home", plan_frequency: int=10,
                              init_qpos=None, init_qvel=None, seed=None) -> None:
        """
        Generate long trajectory using receding horizon MPC.
        
        This method significantly reduces gRPC communication overhead by only
        re-planning every N timesteps instead of at every simulation step.
        
        Args:
            keyframe: Initial keyframe to start from
            plan_frequency: Re-plan every N timesteps (reduces gRPC calls)
            init_qpos: Optional initial joint positions (overrides keyframe)
            init_qvel: Optional initial joint velocities (overrides keyframe)
            seed: Optional random seed for state noise initialization (used when init_state_noise_flag=True)
        """
        # Reset data
        mujoco.mj_resetData(self.model, self.data)
        
        # Reset to specified keyframe
        keyframe_id = mujoco.mj_name2id(self.model,
                                        mujoco.mjtObj.mjOBJ_KEY,
                                        keyframe)
        if keyframe_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, keyframe_id)
        else:
            if self.verbose > 0:
                print(f"Warning: '{keyframe}' keyframe not found in model")
        
        # Apply initialization noise if requested
        if self.init_state_noise_flag:
            # Set random seed for reproducibility
            if seed is not None:
                np.random.seed(seed)
            
            # For Shadow task, use custom randomization
            if self.task_id == "Shadow":
                self._randomize_shadow_cube_and_goal()
            else:
                # For other tasks, use standard qpos/qvel noise
                self.qpos_noise = np.random.uniform(self.qpos_noise_rnge[0],
                                                    self.qpos_noise_rnge[1],
                                                    size=self.model.nq)
                self.qvel_noise = np.random.uniform(self.qvel_noise_rnge[0],
                                                    self.qvel_noise_rnge[1],
                                                    size=self.model.nv)
                self.data.qpos[:] += self.qpos_noise
                self.data.qvel[:] += self.qvel_noise
        
        # If initial qpos/qvel are provided, override the keyframe state
        if init_qpos is not None:
            self.data.qpos[:] = init_qpos
        if init_qvel is not None:
            self.data.qvel[:] = init_qvel
        
        # Cache initial state
        self.qpos[:, 0] = self.data.qpos
        self.qvel[:, 0] = self.data.qvel
        self.time[0] = self.data.time
        
        # Pre-allocate for planned actions
        planned_actions = None
        
        for t in range(self.rollout_horizon - 1):
            if self.verbose > 0 and t % 100 == 0:
                print(f"Planning step {t}/{self.rollout_horizon}")
            
            # Only re-plan every N steps (instead of every step!)
            if t % plan_frequency == 0:
                # gRPC call: Update state
                self.agent.set_state(
                    time=self.data.time,
                    qpos=self.data.qpos,
                    qvel=self.data.qvel,
                    act=self.data.act,
                    mocap_pos=self.data.mocap_pos,
                    mocap_quat=self.data.mocap_quat,
                    userdata=self.data.userdata
                )
                
                # gRPC call: Run optimization
                for _ in range(self.opt_steps):
                    self.agent.planner_step()
                
                # gRPC call: Get best trajectory
                trajectory = self.agent.best_trajectory()
                planned_actions = trajectory['actions']  # (horizon_steps, nu)
                
                if self.verbose > 1:
                    print(f"  Re-planned at step {t}, got {len(planned_actions)} actions")
            
            # Use cached action from the planned trajectory
            # The action index within the current planning window
            action_idx = t % plan_frequency
            
            # Clamp to last available action if we're at the end of planned horizon
            action_idx = min(action_idx, len(planned_actions) - 1)
            
            # Set control from planned trajectory (no gRPC!)
            self.data.ctrl = planned_actions[action_idx]
            self.ctrl[:, t] = self.data.ctrl
            
            # Step simulation (no gRPC!)
            mujoco.mj_step(self.model, self.data)
            
            # Cache states (no gRPC!)
            self.qpos[:, t+1] = self.data.qpos
            self.qvel[:, t+1] = self.data.qvel
            self.time[t+1] = self.data.time
        
        # Reset the agent
        self.agent.reset()

    def set_rollout_horizon(self, rollout_horizon: int):
        self.rollout_horizon = rollout_horizon

    def get_rollout_horizon(self) -> int:
        return self.rollout_horizon
    
    def set_init_state_noise_flag(self, init_state_noise_flag: bool):
        self.init_state_noise_flag = init_state_noise_flag
    
    def get_init_state_noise_flag(self) -> bool:
        return self.init_state_noise_flag
    
    def set_qpos_noise_range(self, qpos_noise_rnge: tuple[float, float]):
        self.qpos_noise_rnge = qpos_noise_rnge
    
    def get_qpos_noise_range(self) -> tuple[float, float]:
        return self.qpos_noise_rnge
    
    def set_qvel_noise_range(self, qvel_noise_rnge: tuple[float, float]):
        self.qvel_noise_rnge = qvel_noise_rnge
    
    def get_qvel_noise_range(self) -> tuple[float, float]:
        return self.qvel_noise_rnge
    
    def get_trajectories(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.qpos, self.qvel, self.ctrl, self.time
    
    def get_costs(self) -> tuple[np.ndarray, np.ndarray]:
        return self.cost_total, self.cost_terms
    
    def get_ctrl_downsampled(self, downsample_factor: int = None) -> np.ndarray:
        """
        Get downsampled control trajectory for comparison with RL environments.
        
        The MPC ctrl array is stored at physics_timestep resolution (e.g., 0.0025s),
        but RL environments typically expect actions at a coarser control_timestep
        (e.g., 0.025s for walker). This method downsamples by taking every Nth control.
        
        If downsample_factor is not provided, it automatically uses the ratio
        between agent_timestep and physics_timestep.
        
        Args:
            downsample_factor: Number of physics steps per control action.
                             If None, uses self.steps_per_agent_update.
        
        Returns:
            Downsampled control array with shape (nu, rollout_horizon // downsample_factor)
        """
        if downsample_factor is None:
            downsample_factor = self.steps_per_agent_update
        return self.ctrl[:, ::downsample_factor]
