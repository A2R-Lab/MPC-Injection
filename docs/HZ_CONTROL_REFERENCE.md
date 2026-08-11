# Policy Hz, PD Hz, and Unitree Low-Level Control

This note summarizes the Hz-related discussion for the MPC-RL Go2 training and deployment stack.

## Current Timing Model

There are three different update rates to keep separate:

| Layer | Where | Current rate | What it does |
|---|---|---:|---|
| RL policy in training | `mpc_rl/envs/velocity_tracking_env.py` | 50 Hz | Chooses a new 12D action / joint target residual |
| Simulated PD in training | `mpc_rl/envs/velocity_tracking_env.py` | 200 Hz | Recomputes torques at each MuJoCo physics step while holding the latest policy target |
| Go2 barrel-roll policy | `mpc_rl/envs/barrel_roll_env.py` | 50 Hz for 2.50 s | Runs 125 policy steps per schema-v3 episode |
| Go2 barrel-roll demonstration MPC | `mpc_rl/planner/gen_traj_data_barrel_roll.py` | 50 Hz replanning, 100 Hz plan nodes | Replans only through the 1.40 s maneuver, then holds the terminal-padded final-stance plan through 2.50 s |
| Go2 barrel-roll demonstration plant | `mpc_rl/planner/gen_traj_data_barrel_roll.py` | 200 Hz | Applies direct MPC/tracking torque for 500 MuJoCo physics steps |
| RL policy in deployment | `deploy/robots/go2/config/policy/velocity/v0/params/deploy.yaml` | 50 Hz | Runs ONNX and updates desired joint positions |
| Low command publishing in deployment | `deploy/include/FSM/CtrlFSM.h` and `deploy/robots/go2/src/State_RLBase.cpp` | 1 kHz | Rewrites the latest desired joint position into `LowCmd` |
| Actual motor-side PD on robot | Unitree firmware / motor controller | Not directly set here | Tracks `q`, `dq`, `kp`, `kd`, and `tau` from `LowCmd` |

## Training Policy Hz

Training policy Hz is controlled by:

```python
sim_dt * decimation
```

in `mpc_rl/envs/velocity_tracking_env.py`.

Current defaults:

```python
sim_dt = 0.005
decimation = 4
control_dt = 0.02  # 50 Hz
```

The policy action is converted once into a raw joint target, passed through the action LPF, then the environment runs the PD loop for `decimation` MuJoCo steps:

```python
raw_q_target = default_joint_pos + action_scale * action
q_target = action_lpf(raw_q_target)
for _ in range(self.decimation):
    torques = self.kp * (q_target - q_current) + self.kd * (0.0 - dq_current)
    mujoco.mj_step(self.mjModel, self.mjData)
```

So with `sim_dt = 0.005` and `decimation = 4`:

- policy update rate: 50 Hz
- simulated PD torque update rate: 200 Hz

The schema-v3 Go2 barrel-roll task uses those same outer and inner rates but a
fixed 2.50 s horizon: 125 policy transitions and 500 physics transitions. Its
MPC maneuver remains 1.40 s long. Demonstration generation stops replanning at
that boundary and repeats the terminal-padded final-stance plan with the same
PD feedback for the remaining 1.10 s.

## Action Low-Pass Filter

The action LPF sits between the policy and the PD controller. The policy still
outputs a 12D residual action in `[-1, 1]`, and that raw policy action is still
what appears in the `last_action` observation. The filter is applied after the
raw action is converted into an absolute joint-position target:

```python
raw_q_target = default_joint_pos + action_scale * action
filtered_q_target += alpha * (raw_q_target - filtered_q_target)
```

The PD controller tracks `filtered_q_target`, not `raw_q_target`:

```python
torques = kp * (filtered_q_target - q_current) + kd * (0.0 - dq_current)
```

This is a first-order exponential low-pass filter. The coefficient is computed
from the cutoff frequency and the policy/control timestep:

```python
alpha = 1.0 - exp(-2.0 * pi * cutoff_hz * control_dt)
```

For the current 50 Hz policy timing and 5 Hz cutoff:

```python
control_dt = 0.02
cutoff_hz = 5.0
alpha = 1.0 - exp(-2.0 * pi * 5.0 * 0.02)  # about 0.467
```

So each policy step moves the commanded joint target about 46.7% of the way
from the previous filtered target toward the new raw target. Sudden target
changes are smoothed, but steady targets are eventually reached.

In training this is configured by `action_lpf_cutoff_hz` in
`mpc_rl/envs/velocity_tracking_env.py`. Set it to `None` or `<= 0` to disable
the filter. In deployment this is configured by `low_pass_filter_cutoff_hz` in:

```text
deploy/robots/go2/config/policy/velocity/v0/params/deploy.yaml
```

The deployment filter uses `step_dt` as its timestep, so changing deployment
policy Hz changes `alpha` for the same cutoff. If you retrain and deploy at a
new policy rate, keep the cutoff frequency consistent between training and
deployment rather than manually matching the old `alpha`.

If you retrain at 25 Hz, the clean change is:

```python
decimation = 8
```

while keeping:

```python
sim_dt = 0.005
```

That gives:

- policy update rate: 25 Hz
- simulated PD torque update rate: still 200 Hz

This matches the hardware structure better than lowering the physics/PD rate, because on the real robot the policy target is slower than the low-level motor tracking.

## Deploy Policy Hz

Deployment policy Hz is controlled by:

```yaml
step_dt: 0.02
```

in:

```text
deploy/robots/go2/config/policy/velocity/v0/params/deploy.yaml
```

The C++ deployment code reads this value in:

```text
deploy/include/isaaclab/envs/manager_based_rl_env.h
```

and uses it to pace the policy thread in:

```text
deploy/include/FSM/State_RLBase.h
```

Current value:

```yaml
step_dt: 0.02  # 50 Hz
```

For 25 Hz deployment:

```yaml
step_dt: 0.04  # 25 Hz
```

The FSM itself still runs at 1 kHz and writes the most recent processed, low-pass-filtered action into `lowcmd->motor_cmd()[...].q()` in:

```text
deploy/robots/go2/src/State_RLBase.cpp
```

So at 25 Hz deployment, the policy updates a desired joint target every 40 ms, and the latest target is held/re-published between policy updates.

## Case 1: Existing 50 Hz Policy Deployed at 25 Hz

Change only:

```yaml
step_dt: 0.04
```

in:

```text
deploy/robots/go2/config/policy/velocity/v0/params/deploy.yaml
```

Do not change:

- ONNX export
- observation order
- action scale
- joint order
- `kp` / `kd`
- `stiffness` / `damping`

Important caveat: this is a distribution shift. A policy trained with 20 ms between actions will now see 40 ms between action updates. Joint velocities, contact timing, and `last_action` dynamics may no longer match training.

## Case 2: Retrain at 25 Hz and Deploy at 25 Hz

Training changes:

```python
sim_dt = 0.005
decimation = 8
```

in:

```text
mpc_rl/envs/velocity_tracking_env.py
```

Deployment change:

```yaml
step_dt: 0.04
```

in:

```text
deploy/robots/go2/config/policy/velocity/v0/params/deploy.yaml
```

Also consider changing time-count parameters so the same real-time durations are preserved:

- `command_resample_interval`: current `250` control steps is 5 seconds at 50 Hz. At 25 Hz, use `125` for 5 seconds.
- `max_episode_steps`: current `1000` control steps is 20 seconds at 50 Hz. At 25 Hz, use `500` for 20 seconds, or leave it at `1000` if you want 40-second episodes.

For ONNX sim testing, `deploy/test_onnx_policy.py` currently creates `QuadrupedVelocityTracking-v0` with default timing. To test a 25 Hz policy faithfully, make that test environment use the same `decimation = 8` timing.

## Unitree Low-Level Controller Access

Public Unitree SDK/ROS2 material indicates that the Go2/B2-style low-level interface exposes motor command fields through `LowCmd`:

- `q`: target joint position
- `dq`: target joint velocity
- `kp`: proportional gain
- `kd`: derivative gain
- `tau`: feedforward torque

The relevant topics are `rt/lowcmd` / `rt/lowstate` in SDK2-style DDS examples, and `/lowcmd` / low-state topics in the ROS2 bridge material.

That means this code already accesses the low-level command interface by writing:

```cpp
lowcmd->msg_.motor_cmd()[motor_id].q() = action[i];
lowcmd->msg_.motor_cmd()[i].kp() = ...
lowcmd->msg_.motor_cmd()[i].kd() = ...
```

What appears not to be exposed is changing the internal Unitree motor-controller servo frequency itself. The public interface lets you publish desired position/velocity/torque and gains at your chosen command rate; the embedded controller then tracks those commands internally.

Practical interpretation:

- You can control the outer command/update rate from this code.
- You can set `q`, `dq`, `kp`, `kd`, and `tau`.
- You should not assume you can directly set the internal motor PD loop Hz from this software stack.

## Online References

- Unitree ROS2 README, low-level motor control and `LowCmd` fields: https://github.com/unitreerobotics/unitree_ros2
- Unitree SDK2 Go2 stand example using `rt/lowcmd`, `rt/lowstate`, `q`, `dq`, `kp`, `kd`, `tau`, CRC, and a low-command write loop: https://github.com/unitreerobotics/unitree_sdk2/blob/main/example/go2/go2_stand_example.cpp
- Unitree SDK2 low-level control overview via DeepWiki mirror of the public repo: https://deepwiki.com/unitreerobotics/unitree_sdk2/7-low-level-control
- Unitree SDK2 Python low-level control overview via DeepWiki mirror: https://deepwiki.com/unitreerobotics/unitree_sdk2_python/3-low-level-control
