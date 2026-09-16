# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""DM Control loading with the paper's velocity-only walker reward.

The task inherits upstream physics, observations and initialization. This
replaces the reward edit formerly made in the installed dm_control package.
"""

from dm_control import suite
from dm_control.rl import control
from dm_control.suite import walker
from dm_control.utils import rewards


class VelocityWalker(walker.PlanarWalker):
    """Preserve the research walker reward; standing remains upstream."""

    def get_reward(self, physics):
        if self._move_speed == 0:
            return super().get_reward(physics)
        move_reward = rewards.tolerance(
            physics.horizontal_velocity(),
            bounds=(self._move_speed, float("inf")),
            margin=self._move_speed / 2,
            value_at_margin=0.5,
            sigmoid="linear",
        )
        return (5 * move_reward + 1) / 6


def _walker_env(move_speed, time_limit=25, random=None, environment_kwargs=None):
    physics = walker.Physics.from_xml_string(*walker.get_model_and_assets())
    task = VelocityWalker(move_speed=move_speed, random=random)
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=0.025,
        **(environment_kwargs or {}),
    )


def load_dm_control_env(domain_name, task_name, task_kwargs=None,
                        environment_kwargs=None, visualize_reward=False):
    """Load the local walker task, or delegate other tasks to DM Control."""
    speeds = {"stand": 0, "walk": 1, "run": 8}
    if domain_name != "walker" or task_name not in speeds:
        return suite.load(domain_name, task_name, task_kwargs,
                          environment_kwargs, visualize_reward)
    kwargs = dict(task_kwargs or {})
    if environment_kwargs is not None:
        kwargs["environment_kwargs"] = environment_kwargs
    env = _walker_env(speeds[task_name], **kwargs)
    env.task.visualize_reward = visualize_reward
    return env
