# walker_fixed_init.py
import gymnasium as gym
import numpy as np
import shimmy  # registers dm_control/* with Gymnasium

def set_xml_initial_state(env):
    """Force qpos/qvel to the model's XML defaults after every reset."""
    physics = env.unwrapped._env.physics
    # Write initial configuration inside a reset context, then forward.
    with physics.reset_context():
        physics.data.qpos[:] = physics.model.qpos0
        physics.data.qvel[:] = 0
        print(physics.model.qpos0)
    physics.forward()

def main():
    # Disable dm_control's episode randomization at source.
    env = gym.make(
        "dm_control/walker-walk-v0",
        task_kwargs={"random": None},          # no per-episode randomness
        render_mode="human",
    )

    obs, info = env.reset(seed=0)
    set_xml_initial_state(env)

    for _ in range(1000):
        action = env.action_space.sample()*0
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
            set_xml_initial_state(env)

    env.close()

if __name__ == "__main__":
    main()

