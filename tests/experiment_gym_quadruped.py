from gym_quadruped.quadruped_env import QuadrupedEnv
import mpc_rl.envs

def test_gym_quadruped():
    robot_name = "go2"
    scene_name = "flat"
    state_observables_names = tuple(QuadrupedEnv.ALL_OBS) # return all available state observables

    env = QuadrupedEnv(robot=robot_name,
                    scene=scene_name,
                    base_vel_command_type="human", # "forward", "random", "forward+rotate", "human"
                    state_obs_names=state_observables_names, # Desired quantities in the "state"
                    )
    obs = env.reset()

    env.render()

    for _ in range(10000):
        action = env.action_space.sample() * 10 # sample random action
        state, reward, is_terminated, is_truncated, info = env.step(action=action)

        if is_terminated:
            pass
            # Do some stuff

        env.render()

    env.close()

def test_gym_quadruped_velocity_tracking():
    env = mpc_rl.envs.QuadrupedVelocityTrackingEnv(
        robot="go2", 
        scene="flat",
        render_mode="human"  # Enable human-visible rendering
    )
    obs = env.reset()
    env.render()
    for _ in range(10000):
        action = env.action_space.sample() * 10 # sample random action
        state, reward, is_terminated, is_truncated, info = env.step(action=action)

        if is_terminated:
            pass
            # Do some stuff

        env.render()

    env.close()

if __name__ == "__main__":
    #test_gym_quadruped()
    test_gym_quadruped_velocity_tracking()