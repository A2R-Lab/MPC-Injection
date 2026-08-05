# Low-Pass Filtering RL Action Targets

This note describes a reusable pattern for adding a low-pass filter (LPF) to an
RL control pipeline. The example integration is the velocity-tracking policy in
this repository, but the same design applies to other learned controllers that
emit actuator targets.

## Purpose

Use an action LPF to smooth the physical command sent to a downstream actuator
controller. In this pattern the policy still emits its normal action, but the
runtime wrapper converts that action into a physical target and filters that
target before the PD controller or hardware interface sees it.

The filter reduces high-frequency target jumps that can produce torque spikes,
joint chatter, or sim-to-real mismatch. It is not an observation filter, reward
term, or replacement for actuator limits.

## Filter Model

The implementation uses a first-order discrete filter:

```text
y[t] = y[t-1] + alpha * (x[t] - y[t-1])
alpha = 1 - exp(-2 * pi * cutoff_hz * dt)
```

Where:

- `x[t]` is the new unfiltered physical target.
- `y[t]` is the filtered physical target sent onward.
- `dt` is the action update period, not necessarily the physics substep.
- `cutoff_hz <= 0` or `null` disables filtering.

For the Go2 velocity policy, `cutoff_hz = 5.0` and `dt = 0.02`, so
`alpha ~= 0.467`.

## Integration Pattern

Place the LPF after policy-output postprocessing and before actuator control:

```text
policy_action in [-1, 1]
raw_target = postprocess(policy_action)  # scale, offset, optional clipping
filtered_target = low_pass_filter(raw_target)
actuator_controller(filtered_target)
```

Prefer filtering absolute physical targets, not normalized network outputs. This
keeps training and deployment behavior tied to the actual command sent to the
robot, and it avoids ambiguity when scale, offset, clipping, or joint ordering
changes outside the policy.

## Training Example

`mpc_rl/envs/velocity_tracking_env.py` filters joint-position targets before the
MuJoCo PD loop:

- `action_lpf_cutoff_hz` configures the cutoff; the default is `5.0`.
- `control_dt = sim_dt * decimation` is the filter timestep. With defaults,
  physics runs at 200 Hz and actions update at 50 Hz.
- `_compute_lpf_alpha()` converts cutoff and `control_dt` into `alpha`.
- On `reset()`, `_filtered_q_target` is set to `default_joint_pos`.
- On each `step()`, the raw action is clipped to `[-1, 1]`, converted to an
  absolute joint target, filtered, then held constant through all inner MuJoCo
  physics substeps:

```text
raw_q_target = default_joint_pos + action_scale * action
q_target = action_lpf(raw_q_target)
torque = kp * (q_target - q) + kd * (0 - dq)
```

If filtering is disabled, `alpha` is treated as `1.0` and the raw target passes
through unchanged.

## Deployment Example

The deployment code mirrors the same action-target filter in C++:

- `deploy/robots/go2/config/policy/velocity/v0/params/deploy.yaml` defines the
  action `scale`, `offset`, and `low_pass_filter_cutoff_hz: 5.0`.
- `deploy/include/isaaclab/envs/mdp/actions/joint_actions.h` computes `alpha`
  from `low_pass_filter_cutoff_hz` and `env->step_dt`.
- `JointAction::process_actions()` applies scale, offset, optional clipping,
  then filters `_processed_actions`.
- `ManagerBasedRLEnv::step()` runs the ONNX policy and updates the filtered
  action at `step_dt = 0.02`.
- `State_RLBase::run()` publishes the latest filtered joint targets to Unitree
  `LowCmd` motor position fields; the robot-side PD gains are configured from
  the same deploy YAML.

Deployment reset matters: `JointAction::reset()` processes a zero policy action,
which becomes the default joint pose after offsetting, and initializes the LPF
state there. The first live policy output is therefore filtered from the neutral
standing target rather than from all-zero joint positions.

## Design Requirements

When adding this pattern to another RL pipeline:

1. Match training and deployment semantics. Filter the same quantity in the same
   location of the action pipeline.
2. Use the actual action update period when computing `alpha`. If deployment
   updates targets at a different rate than training, recompute `alpha` for that
   rate or expect different smoothing.
3. Initialize filter state at reset/startup to a safe target, usually the
   default pose or current commanded target.
4. Apply actuator limits independently. The LPF smooths changes; it does not
   guarantee targets, velocities, or torques are safe.
5. Log raw and filtered targets during bring-up. A mismatch here is easier to
   diagnose than a downstream gait or hardware symptom.

## Tuning Notes

Lower cutoff values smooth more but add more lag. Higher cutoff values track the
policy more directly but allow sharper target changes. Start with the same
cutoff in simulation and deployment, then tune only if hardware traces show the
filtered target is either too sluggish or still too sharp.
