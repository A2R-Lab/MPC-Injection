"""Gymnasium-compatible three-legged cheetah environment.

The task is based on `dm_control.suite.cheetah`, but uses the local
three-legged cheetah XML and the same initial-state randomization as
`mpc_rl/planner/gen_traj_data_cheetah3.py`'s default `walker_like` mode.
"""

from __future__ import annotations

import collections
from pathlib import Path
from typing import Any

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.utils import rewards
import numpy as np
from shimmy import DmControlCompatibilityV0

from mpc_rl.envs.cheetah3_common import sample_valid_cheetah3_initial_state


CHEETAH3_MODEL_PATH = Path(__file__).parent.parent / "tasks/cheetah/task.xml"
DEFAULT_SPEED_GOAL = 2.0
DEFAULT_TIME_LIMIT = 10.0


class Physics(mujoco.Physics):
    """Physics helpers for the three-legged cheetah."""

    def speed(self) -> float:
        """Return horizontal torso subtree speed."""
        return float(self.named.data.sensordata["torso_subtreelinvel"][0])


class Cheetah3Task(base.Task):
    """Run task for the three-legged cheetah."""

    def __init__(self, speed_goal: float = DEFAULT_SPEED_GOAL, random=None):
        super().__init__(random=random)
        self._speed_goal = float(speed_goal)

    @property
    def speed_goal(self) -> float:
        return self._speed_goal

    def initialize_episode(self, physics: Physics):
        """Randomize the initial state like cheetah3 trajectory generation."""
        if physics.model.nq != physics.model.njnt:
            raise ValueError("cheetah3 initialization assumes one qpos per joint.")

        # Match gen_traj_data_cheetah3.py `walker_like`, but reject poses that
        # start upside down or with any non-ground geom below the floor.
        qpos, qvel = sample_valid_cheetah3_initial_state(
            physics.model.ptr,
            self.random,
        )
        physics.data.qpos[:] = qpos
        physics.data.qvel[:] = qvel

        physics.data.time = 0.0
        physics.forward()
        super().initialize_episode(physics)

    def get_observation(self, physics: Physics):
        """Return the original cheetah observation: position without x, velocity."""
        obs = collections.OrderedDict()
        obs["position"] = physics.data.qpos[1:].copy()
        obs["velocity"] = physics.velocity()
        return obs

    def get_reward(self, physics: Physics) -> float:
        """Reward only forward speed relative to the configured speed goal."""
        return float(
            rewards.tolerance(
                physics.speed(),
                bounds=(self._speed_goal, float("inf")),
                margin=self._speed_goal,
                value_at_margin=0.0,
                sigmoid="linear",
            )
        )


def make_dm_control_env(
    speed_goal: float = DEFAULT_SPEED_GOAL,
    time_limit: float = DEFAULT_TIME_LIMIT,
    random=None,
) -> control.Environment:
    """Create the underlying DM Control cheetah3 environment."""
    physics = Physics.from_xml_path(str(CHEETAH3_MODEL_PATH))
    task = Cheetah3Task(speed_goal=speed_goal, random=random)
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        n_sub_steps=1,
    )


class Cheetah3Env(DmControlCompatibilityV0):
    """Shimmy Gymnasium wrapper for the local DM Control-style cheetah3 task."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(
        self,
        render_mode: str | None = None,
        speed_goal: float = DEFAULT_SPEED_GOAL,
        time_limit: float = DEFAULT_TIME_LIMIT,
        render_kwargs: dict[str, Any] | None = None,
    ):
        self.speed_goal = float(speed_goal)
        self.time_limit = float(time_limit)
        dm_env = make_dm_control_env(
            speed_goal=self.speed_goal,
            time_limit=self.time_limit,
        )
        super().__init__(
            dm_env,
            render_mode=render_mode,
            render_kwargs=render_kwargs,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        if seed is not None:
            self._env.task._random = np.random.RandomState(seed)
        return super().reset(seed=seed, options=options)


def make_cheetah3_env(
    render_mode: str | None = None,
    speed_goal: float = DEFAULT_SPEED_GOAL,
    time_limit: float = DEFAULT_TIME_LIMIT,
) -> Cheetah3Env:
    """Entry point used by gymnasium registration and training code."""
    return Cheetah3Env(
        render_mode=render_mode,
        speed_goal=speed_goal,
        time_limit=time_limit,
    )
