from gym_quadruped.quadruped_env import QuadrupedEnv

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
    action = env.action_space.sample() * 100 # sample random action
    state, reward, is_terminated, is_truncated, info = env.step(action=action)

    if is_terminated:
        pass
        # Do some stuff

    env.render()

env.close()