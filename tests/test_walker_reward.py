"""Guard the paper reward and unchanged DM Control transition behavior."""
import numpy as np
import pytest
from dm_control import suite

from mpc_rl.envs.dm_control_env import VelocityWalker, load_dm_control_env


@pytest.mark.parametrize("speed,expected", [(-1, 1 / 6), (0, 1 / 6), (0.5, 7 / 12), (1, 1), (2, 1)])
def test_walking_reward_ignores_posture(speed, expected):
    class FallenWalker:
        def horizontal_velocity(self):
            return speed

        def torso_height(self):
            return 0.1

        def torso_upright(self):
            return -1.0

    assert VelocityWalker(1).get_reward(FallenWalker()) == pytest.approx(expected)


@pytest.mark.parametrize("task", ["stand", "walk", "run"])
def test_walker_preserves_upstream_transitions(task):
    local = load_dm_control_env("walker", task, task_kwargs={"random": 7})
    upstream = suite.load("walker", task, task_kwargs={"random": 7})
    rng = np.random.RandomState(11)
    try:
        a, b = local.reset(), upstream.reset()
        for _ in range(5):
            for key in a.observation:
                np.testing.assert_array_equal(a.observation[key], b.observation[key])
            np.testing.assert_array_equal(local.physics.get_state(), upstream.physics.get_state())
            if task == "stand":
                assert a.reward == b.reward
            action = rng.uniform(-1, 1, local.action_spec().shape)
            a, b = local.step(action), upstream.step(action)
    finally:
        local.close()
        upstream.close()
