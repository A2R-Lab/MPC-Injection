# Policy Hz, PD Hz, and Unitree Low-Level Control

This note summarizes the Hz-related discussion for the MPC-RL Go2 training and deployment stack.

## Current Timing Model

There are three different update rates to keep separate:

| Layer | Where | Current rate | What it does |
|---|---|---:|---|
| RL policy in training | `mpc_rl/envs/velocity_tracking_env.py` | 50 Hz | Chooses a new 12D action / joint target residual |
| Simulated PD in training | `mpc_rl/envs/velocity_tracking_env.py` | 200 Hz | Recomputes torques at each MuJoCo physics step while holding the latest policy target |
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

The policy action is converted once into `q_target`, then the environment runs the PD loop for `decimation` MuJoCo steps:

```python
for _ in range(self.decimation):
    torques = self.kp * (q_target - q_current) + self.kd * (0.0 - dq_current)
    mujoco.mj_step(self.mjModel, self.mjData)
```

So with `sim_dt = 0.005` and `decimation = 4`:

- policy update rate: 50 Hz
- simulated PD torque update rate: 200 Hz

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

The FSM itself still runs at 1 kHz and writes the most recent processed action into `lowcmd->motor_cmd()[...].q()` in:

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
