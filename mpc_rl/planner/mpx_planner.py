import jax.numpy as jnp
import jax
import mujoco
# Update JAX configuration
jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

import numpy as np
from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.utils.mujoco.visual import render_sphere, render_vector

import mpx.utils.mpc_wrapper as mpc_wrapper
import mpx.config.config_go2 as config

from timeit import default_timer as timer
# Set GPU device for JAX
gpu_device = jax.devices('gpu')[0]
jax.default_device(gpu_device)

class MPXPlanner():
    """
    MPC planner using MPX (Model Predictive Control in JaX) for legged robot MPC with the trajectory
    optimization done in JAX. Note that the controller uses MJX and JAX, but the environment uses
    MuJoCo. This planner is designed to collect the trajectories from a run of the environment.

    Default configuration for this planner is for the Unitree Go2 quadruped.
    """

    def __init__(self,
               episode_length: int = 1000 # Default same as the gymnasium environment QuadrupedVelocityTrackingEnv
        ) -> None:
        """
        Args:
            rollout_horizon: Maximum number of steps to run the environment and collect trajectories.
        
        TODO: Figure out how to handle the random initialization.
        TODO: Double check sim_frequency b/w QuadrupedEnv and QuadrupedVelocityTrackingEnv.
        TODO: Check how to vary the base_vel_command_type to be randomized like in QuadrupedVelocityTrackingEnv.

        """

        # Define robot and scene parameters
        self.robot_name = "go2"
        self.scene_name = "flat"
        self.robot_feet_geom_names = dict(FR='FR', FL='FL', RR='RR', RL='RL')
        self.robot_leg_joints = dict(FR=['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', ],
                                     FL=['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', ],
                                     RR=['RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint', ],
                                     RL=['RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint'])
        self.state_observables_names = tuple(QuadrupedEnv.ALL_OBS) # Return all available state observables

        # Initialize simulation environment
        self.episode_length = episode_length
        self.sim_frequency = 200.0
        self.env = QuadrupedEnv(robot=self.robot_name,
                                scene=self.scene_name,
                                sim_dt=1 / self.sim_frequency,  # Simulation time step [s]
                                ref_base_lin_vel=0.0,  # Constant magnitude of reference base linear velocity [m/s]
                                ground_friction_coeff=0.7,  # pass a float for a fixed value
                                base_vel_command_type="human",  # "forward", "random", "forward+rotate", "human"
                                state_obs_names=self.state_observables_names,  # Desired quantities in the 'state'
                                )
        self.obs = self.env.reset(random=False)
        self.counter = 0 # Simulation step counter

        # Define the MPC wrapper
        self.mpc_frequency = config.mpc_frequency
        self.mpc = mpc_wrapper.MPCControllerWrapper(config)
        self.env.mjData.qpos = jnp.concatenate([config.p0, config.quat0, config.q0])
        self.env.render()
        self.tau = jnp.zeros(config.n_joints)
        self.tau_old = jnp.zeros(config.n_joints)
        self.delay = int(0.007 * self.sim_frequency) # TODO: Determine if we even need this
        self.q = config.q0.copy()
        self.dq = jnp.zeros(config.n_joints)
        self.mpc_time = 0
        self.mpc.robot_height = config.robot_height
        self.mpc.reset(self.env.mjData.qpos.copy(), self.env.mjData.qvel.copy())

        # Save trajectories
        self.qpos_traj = []
        self.qvel_traj = []
        self.tau_traj = []

    def plan_and_sim(self) -> None:
        """
        Main loop to run the environment and collect trajectories using the MPC planner.

        TODO: Figure out how long to run a sim for.
        TODO: Figure out how to initialize state randomly in same range as the gym env
        """
        for t in range(self.episode_length):
            qpos = self.env.mjData.qpos.copy()
            qvel = self.env.mjData.qvel.copy()

            if (self.counter % (self.sim_frequency / self.mpc_frequency) == 0 or self.counter == 0):
                ref_base_lin_vel = self.env._ref_base_lin_vel_H
                ref_base_ang_vel = np.array([0., 0., self.env._ref_base_ang_yaw_dot])

                # TODO: Figure out how to have automatic random commands here like in QuadrupedVelocityTrackingEnv
                input = np.array([ref_base_lin_vel[0], ref_base_lin_vel[1], ref_base_lin_vel[2],
                                  ref_base_ang_vel[0], ref_base_ang_vel[1], ref_base_ang_vel[2],
                                  config.robot_height])
                
                contact_temp, _ = self.env.feet_contact_state()
                contact = np.array([contact_temp[self.robot_feet_geom_names[leg]] for leg in ['FL', 'FR', 'RL', 'RR']])

                if self.counter != 0:
                    for i in range(self.delay):
                        # TODO: Do we really need to read the state again during the delay? Or can we just use the state from before the delay?
                        qpos = self.env.mjData.qpos.copy()
                        qvel = self.env.mjData.qvel.copy()

                        # PD feedback torque term
                        tau_fb = 10 * (self.q - qpos[7:7 + config.n_joints]) - 2 * (qvel[6:6 + config.n_joints])
                        state, reward, is_terminated, is_truncated, info = self.env.step(action=self.tau + tau_fb)
                        self.counter += 1

                start = timer()
                self.tau, self.q, self.dq = self.mpc.run(qpos, qvel, input, contact)
                stop = timer()
                # print("Time taken for MPC: ", stop - start)

            tau_fb = 10 * (self.q - qpos[7:7 + config.n_joints]) - 2 * (qvel[6:6 + config.n_joints])
            state, reward, is_terminated, is_truncated, info = self.env.step(action=self.tau + tau_fb)

            self.counter += 1
            self.env.render()



if __name__ == "__main__":
    # Example usage of the MPXPlanner
    print("Running...")
    planner = MPXPlanner(episode_length=1000)
    planner.plan_and_sim()
    print("Done!")