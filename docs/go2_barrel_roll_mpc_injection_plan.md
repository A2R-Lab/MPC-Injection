# Go2 Barrel Roll MPC-Injection Implementation Plan

## Status and purpose

This document is the implementation and experiment plan for a simulation-first
Go2 barrel-roll task in MPC-RL. It is intended to be the durable handoff for
future coding agents and human contributors. It records the decisions made
during the design interview, the repository evidence behind the design, the
required file changes, validation gates, runnable workflows to add, and the
fallback order if a gate fails.

This is a plan, not an implementation report. No barrel-roll environment,
generator, dataset, or trained policy exists in the root project yet.

The coordinated feature branches already exist in the current worktree:

- root: `feature/go2-barrel-roll`, created from
  `4b9783af3f43318fb1d634e07c1bd1358a2db2a3`;
- `deps/mpx`: `feature/go2-barrel-roll`, created from
  `249c7323ab0cd13bea2ddb8ed5f252eb9ccde85c`.

The MPX branch currently contains the user's uncommitted Go2 barrel-roll config
changes and the `barel_roll.py` to `barrel_roll.py` rename. Preserve them. The
untracked tree under
`deploy/robots/go2/config/policy/velocity/policies/` is unrelated user data and
must not be modified, staged, or deleted.

## Goal and definition of done

Build a minimal research prototype in which SAC-MPC learns a single, fixed-
direction, one-shot barrel roll from MPX demonstrations in MuJoCo simulation.

The work is done only when all of the following are true:

1. MPX uses the Go2 model, pose, feet, and joint ordering throughout the
   barrel-roll path; no Aliengo task values remain active.
2. A seeded generator re-solves MPX for each sampled Go2 start stance and then
   executes the solution in the same Gym-Quadruped-based plant used by the RL
   task.
3. Only executed rollouts that pass the frozen barrel-roll success classifier
   are saved as versioned, direct RL transitions.
4. The percentage injector strictly validates and injects the new 45D-policy,
   4D-privileged, 12D-action schema into `TaggedDictReplayBuffer`.
5. `mpc_rl/train.py` can train and evaluate
   `--env_name=quadruped-barrel_roll` with SAC-MPC, 25% MPC replay, no domain
   randomization, and no Go2 sysID patch.
6. A 100,000-step pilot shows increasing held-out barrel-roll success before
   any production training starts.
7. Three 500,000-step production seeds complete, and the median policy success
   rate is at least 80% over 100 held-out initialization seeds per policy. All
   three results are reported; weak seeds are not discarded.
8. Tests, commands, dataset provenance, and user-visible workflow documentation
   are current.

Pure SAC, TD3-MPC, bidirectional rolls, domain randomization, system
identification, and hardware deployment are not required for this MVP.

## Locked design decisions

The following decisions came from the design interview and should not be
silently revisited during implementation:

| Area | Decision |
| --- | --- |
| Robot | Go2 only; prefer Go2 values anywhere an Aliengo/Go2 choice appears. |
| Maneuver | One fixed-direction, one-shot barrel roll from standing. |
| Schedule | Reuse the current 1.0 s MPX schedule unchanged for the MVP. |
| Timing | 200 Hz MuJoCo physics, 50 Hz RL action/observation transitions. |
| Action | Existing 12D joint-position residual action and PD interface. |
| MPC data semantics | Follow the existing direct-transition quadruped pipeline: execute MPX-plus-feedback torques directly, derive the saved residual action by inverse PD, and do not run MPC generation through the action LPF. |
| Actor observation | Keep 45 values; replace the 3 velocity-command slots with normalized phase, `sin(desired_roll)`, and `cos(desired_roll)`. |
| Critic observation | 4 privileged values: body-frame base linear velocity plus unwrapped signed roll progress. A 9D variant adding base height and four contacts is a fallback only if training fails. |
| Reward | Phase-indexed unwrapped-roll tracking, signed roll progress, and terminal success/failure only. Simplification can be studied after the MVP works. |
| Reset variation | One symmetric hip-spread scalar sampled uniformly from 0.00 to 0.10 rad; all other initial state values fixed initially. |
| Robustness | No DR, observation noise, pushes, encoder bias, or sysID patch. |
| RL algorithm | SAC-MPC only for the MVP. |
| Injection | Percentage injection at 25%; use the direct transition path. |
| Data scale | Gates at 10, 100, and 1,000 accepted trajectories. |
| Training | 100k-step one-seed pilot, then 500k steps for each of three seeds. |
| Comparison | A pure-RL/0% baseline is deferred and is not an MVP gate. |
| Workspace | Coordinated root and MPX feature branches in the current worktree. |

## Verified repository baseline

The implementation should extend the accepted Go2 velocity path, not introduce
a third quadruped training stack:

```text
MPX offline Go2 solution per seed
  -> execute torques in a Gym-Quadruped-based Go2 task plant
  -> save 50 Hz direct Dict-observation transitions
  -> strict barrel-roll dataset validation
  -> PercentMPCInjectCallback
  -> TaggedDictReplayBuffer
  -> SB3 SAC-MPC with asymmetric actor/critic inputs
```

Relevant current behavior:

- `deps/mpx/mpx/examples/barrel_roll.py` calls
  `MPCControllerWrapper.runOffline()` and visualizes the optimized `X` sequence
  by assigning `qpos` and `qvel`. It does not execute `U` in the Go2 RL plant.
- `deps/mpx/mpx/config/config_barrel_roll.py` now points at
  `data/go2/go2_mjx.xml` and uses the Go2 `q0` and `p_legs0`, but it still
  contains stale Aliengo labels and an Aliengo-valued `q0_init` declaration.
- `deps/mpx/mpx/utils/mpc_utils.py::reference_barell_roll` defines the current
  one-second reference. Preserve the misspelled internal function name during
  the MVP unless renaming can be proven isolated; the public example filename
  has already been corrected.
- `mpc_rl/envs/velocity_tracking_env.py` already provides the Go2 model loader,
  200/50 Hz stepping, 12D position-residual action, PD loop, Dict observations,
  contacts, rendering, and reset bookkeeping needed by the new task.
- `mpc_rl/planner/gen_traj_data_mpx_dr.py` is the canonical implementation
  pattern for direct quadruped transitions and multi-attempt manifests. Its
  rollout applies controller torques directly and derives an inverse-PD action;
  it does not call the environment action LPF during MPC generation.
- `mpc_rl/common/mpc_inject_callbacks.py::PercentMPCInjectCallback` is the
  production quadruped injection path. It already batches distinct direct
  transitions correctly across vector-environment slots.
- The asymmetric policy feature extractors derive observation widths from the
  Gym spaces. A 4D privileged vector therefore does not require a new policy
  class, although architecture tests and stale 45+3 docstrings need updating.
- `mpc_rl/train.py` currently routes every `quadruped-*` name to
  `QuadrupedVelocityTrackingEnv`, selects the velocity simplified reward for
  MPC algorithms, enables quadruped DR/sysID through global flags, and records
  velocity-specific evaluation videos. These branches must become task-aware.

The orientation handoff recorded 77 passing, 3 failing, and 1 skipped test in
the existing targeted quadruped suite. Those three known failures concern
unrelated stale expectations and are not part of this project. Before code
changes, rerun the same command and record the baseline so this feature does not
introduce additional failures.

## Task contract

### Fixed schedule

Keep the current MPX reference timing as the source schedule:

| Time | Stage |
| ---: | --- |
| 0.00-0.20 s | Initial stance |
| 0.20-0.40 s | Lateral support and push-off |
| 0.40-0.70 s | Flight |
| 0.70-0.80 s | Landing phase |
| 0.80-1.00 s | Final stance |

The signed desired roll stays at zero through 0.20 s, progresses linearly by
one full turn from 0.20 to 0.80 s, and then stays at the signed `2*pi` target.
Use one named constant for the chosen sign in the root task code, assert that it
matches the MPX reference during validation, and save it in every dataset file.

At 50 Hz, an accepted demonstration contains 50 transitions. At 200 Hz it
contains 200 physics steps. The environment should classify success or
incomplete failure on its 50th transition; the Gym `TimeLimit` remains a safety
wrapper rather than the normal source of episode completion.

### Initial-state sampler

Start from the Go2 home keyframe and sample one scalar:

```text
spread ~ Uniform(0.00, 0.10) rad
FL_hip_joint += spread
RL_hip_joint += spread
FR_hip_joint -= spread
RR_hip_joint -= spread
```

Resolve joints by MuJoCo joint name and `jnt_qposadr`, not assumed array order.
Unit tests must prove the sign convention increases left/right foot separation.
Keep base pose, all other joint positions, and all velocities fixed. Run
`mj_forward` and reject any initial state with a joint-limit violation,
non-finite state, foot penetration, or non-foot ground contact.

The generator and RL environment must call the same root-level sampler with the
same seed semantics. MPX receives the exact sampled `qpos` and zero `qvel` and
is re-solved for that state.

### Roll progress

Do not use wrapped Euler roll as the task state. Start from the reset quaternion,
maintain quaternion sign continuity, extract the relative twist about the Go2
longitudinal x-axis, and unwrap successive increments into a signed scalar.
Unit tests must cover both crossings of the `-pi/pi` boundary and reject a
quaternion sign flip as fake progress.

Use that scalar for reward, privileged observation, success classification,
manifest metrics, and evaluation. Keep current projected gravity and body
angular velocity in the actor observation.

### Success and failure classifier

Freeze the success classifier before the 100-trajectory pilot. Initial
commissioning values are proposed below; they are not verified repository facts
and may be adjusted only against the 10-trajectory controller smoke set, before
held-out acceptance seeds are examined:

- accumulated roll reaches at least `2*pi - 0.35` rad in the required direction;
- final roll and pitch error relative to upright are each at most 0.35 rad;
- base height is at least 0.20 m;
- all four filtered foot contacts and the upright/height conditions hold for 5
  consecutive control steps (0.10 s) before the episode ends;
- no torso or other non-foot robot geom touches the ground at any 200 Hz
  substep;
- all state and control arrays remain finite.

One- or two-foot first touchdown is valid. The classifier evaluates the later
stable stance, not the first landing contact.

An early non-foot ground contact or non-finite state is an immediate failure.
Failure to satisfy success on transition 50 is an `incomplete_roll` terminal
failure. Store a structured failure reason; do not encode every failure as a
generic fall.

### Observation contract

The actor remains 45D:

```text
[0:3]    body angular velocity
[3:6]    projected gravity
[6]      normalized maneuver phase in [0, 1]
[7]      sin(desired signed roll)
[8]      cos(desired signed roll)
[9:21]   joint position relative to Go2 default pose
[21:33]  joint velocity
[33:45]  previous raw residual action
```

Phase saturates at 1 during the final stance. The critic receives the same 45D
policy vector plus a 4D privileged vector:

```text
[0:3] body-frame base linear velocity
[3]   unwrapped signed roll progress
```

If the 100k pilot fails specifically because the critic cannot value touchdown
and stabilization states, the first observation fallback is 9D privileged data:
the 4D vector above plus base height and four filtered contacts. Do not adopt
that fallback without recording evidence from the 4D run.

### Action and control contract

Keep `sim_dt=0.005`, `decimation=4`, the current Go2 per-joint PD gains, the
12D normalized residual action, and initially `action_scale=0.5`. Online RL
actions continue through the existing 5 Hz absolute-target LPF.

For demonstrations, preserve the existing direct quadruped convention:

1. Apply MPX feedforward plus joint-state tracking feedback directly to
   `mjData.ctrl` at physics rate.
2. At each 50 Hz transition boundary, derive the saved residual action from the
   first applied torque with the current inverse-PD identity.
3. Clip the saved action to `[-1, 1]` and record whether and by how much the
   unclipped action exceeded the range.
4. Save the directly realized observation, next observation, reward, and done
   tuple. Do not replay the action through the LPF while generating the MPC
   transition.

This is intentionally the existing pipeline behavior, not a claim that the
saved action reproduces the direct-torque transition through `env.step()`.
Record action clipping and torque saturation prominently. If representability
blocks policy learning, increase `action_scale` first and regenerate all data;
do not mix action scales in one dataset. Changing the LPF or action type is a
later fallback, not part of the initial plan.

### Reward contract

Implement only these terms:

```text
reward = roll_tracking + signed_progress + terminal_outcome
```

- `roll_tracking`: an exponential function of the difference between desired
  unwrapped roll and measured unwrapped roll. It must distinguish an upright
  robot at zero progress from one at `2*pi` progress.
- `signed_progress`: positive for progress in the required direction and
  negative for reverse progress. Normalize against the expected per-step roll
  during the 0.20-0.80 s active interval and clip extreme one-step values.
- `terminal_outcome`: a success bonus on transition 50 when the success
  classifier passes; otherwise a failure penalty. Early invalid/contact failure
  also receives the failure penalty.

Put weights and kernels in a task-specific immutable configuration object.
Select initial values before the 10-trajectory smoke test, log each component,
and freeze them before the 100-trajectory pilot. Do not add gait, velocity,
posture, energy, torque, action-rate, or general survival terms during the MVP.
If training fails, diagnose demonstration quality, action clipping, replay
injection, critic learning, and reward traces in that order before adding reward
terms.

## Dependency-ordered implementation

### Gate G0: preserve and checkpoint the workspace

Files: Git metadata only; no source changes.

1. Confirm the root and MPX feature branches named in this document are active.
2. Review the current MPX diff. Commit only the user's Go2 config corrections
   and example rename on the MPX feature branch; do not include unrelated MPX
   changes.
3. Keep the root submodule dirty until that MPX commit exists, then update the
   root gitlink in a later root commit.
4. Record exact root, MPX, nested solver, Gym-Quadruped, MuJoCo, JAX, and Python
   versions in a baseline note or test output.
5. Run the existing targeted quadruped tests in the `mpc-rl` Conda environment
   and retain the known-failure comparison.

Gate: no new baseline failures, no unrelated files staged, and the exported
policy tree remains untouched.

### Gate G1: make the MPX Go2 barrel-roll definition internally consistent

Likely files:

- `deps/mpx/mpx/config/config_barrel_roll.py`
- `deps/mpx/mpx/utils/mpc_utils.py`
- `deps/mpx/mpx/utils/mpc_wrapper.py`
- `deps/mpx/mpx/examples/barrel_roll.py`
- proposed focused tests under `deps/mpx/tests/`

Work:

1. Remove stale active Aliengo values and comments from the barrel-roll config.
   Set or remove `q0_init` deliberately; it is currently Aliengo-valued and is
   not referenced by the inspected barrel-roll path.
2. Assert the configured model is Go2 and that named contact geoms, calf bodies,
   joint ordering, `q0`, and `p_legs0` exist and have expected dimensions.
3. Keep the current schedule and single roll direction. Expose its phase timing
   and roll sign as readable config constants rather than duplicating anonymous
   numbers throughout the generator.
4. Add optional offline-solve diagnostics without breaking the example's
   current four-value unpacking. The generator needs iteration count, final
   objective/constraint norms if available, finite-status, and elapsed time.
5. Ensure all barrel-roll construction passes `use_go2_sysid=False` explicitly.
6. Make the example finite and usable as a smoke command instead of requiring an
   endless viewer loop for validation. Rendering may remain optional.

Validation:

- config/model name and all named IDs resolve;
- reference has `N+1` states, `N` controls, finite values, the intended contact
  stages, one signed turn, and the exact 1.0 s horizon;
- a nominal offline solve returns finite `X` and `U`;
- the user's existing visual result remains intact.

Gate: a finite nominal Go2 plan exists. Direct state visualization alone does
not unlock dataset generation.

### Gate G2: add shared barrel-roll task semantics and the RL environment

Proposed files:

- `mpc_rl/envs/barrel_roll_common.py`
- `mpc_rl/envs/barrel_roll_env.py`
- `mpc_rl/envs/velocity_tracking_env.py` only for small protected step hooks or
  reset parameters needed to avoid duplicating the 200 Hz control loop
- `mpc_rl/envs/__init__.py`
- `tests/test_barrel_roll_env.py`

Design:

1. Put the root task schedule, roll unwrapping, symmetric spread sampler,
   success thresholds, contact classification, and metrics in testable shared
   helpers. Do not import the root package from MPX or the MPX package from the
   environment.
2. Implement `QuadrupedBarrelRollEnv` as a narrow subclass of
   `QuadrupedVelocityTrackingEnv`. Reuse model loading, action conversion, LPF,
   PD stepping, contact IDs, rendering, and base proprioception.
3. Add only small default-no-op protected hooks to the base environment if
   needed to update roll/contact state after each physics substep. Do not copy
   the full `step()` loop into the new class and do not change velocity-task
   defaults.
4. Force Go2, flat terrain, nominal dynamics, disabled DR, disabled pushes/noise,
   and disabled sysID. Reject unsupported robot names rather than silently
   constructing another robot.
5. Override reset to use the shared spread sampler and fully reset phase,
   quaternion continuity, roll progress, landing-contact history, stability
   counter, action/filter state, and failure reason.
6. Override observation, reward, termination, and info semantics according to
   the contracts above. Include `is_success`, `failure_reason`, phase, desired
   roll, roll progress/error, contact state, stability count, and reward
   components in `info`.
7. Register `QuadrupedBarrelRoll-v0` with a 50-step safety limit.

Validation:

- deterministic resets for equal seeds and bounded differing spreads for
  different seeds;
- named hip signs widen the stance;
- observation spaces and values are exactly 45D/4D and finite;
- start, active-roll, and final phase features match the fixed schedule;
- unwrapped roll handles quaternion wrap and sign continuity;
- the reward favors correct signed progress and cannot treat zero-progress
  upright stance as completed `2*pi` progress;
- roll/pitch thresholds used by velocity tracking do not terminate the intended
  inverted phase;
- non-foot contact at any physics substep fails immediately;
- two-foot touchdown is allowed, while stable success requires the later
  four-foot streak;
- velocity-task tests prove unchanged behavior.

Gate: scripted state tests and a simple hand-authored action rollout can traverse
the environment API without NaNs or premature termination.

### Gate G3: execute the offline plan in the Go2 task plant

Proposed file:

- `mpc_rl/planner/gen_traj_data_barrel_roll.py`

Reuse small helpers from `gen_traj_data_mpx_dr.py` where their semantics match;
do not add barrel-roll modes to the DR generator.

Per attempt:

1. Seed the shared spread sampler and construct `QuadrupedBarrelRollEnv` with
   nominal Go2 dynamics, no DR, and no sysID.
2. Copy its exact initial `qpos/qvel` into a shared-process MPX wrapper, call
   `reset()`, and run `runOffline()` for that seed.
3. Treat MPX `X` as the planned state target and `U` as feedforward torque.
   MPX nodes are 100 Hz (`dt=0.01`), so hold each node for two 200 Hz physics
   steps. A 50 Hz RL transition spans two MPX nodes/four physics steps.
4. Follow the accepted quadruped controller convention initially:
   `tau = U + 10*(q_des - q) - 2*dq`, clipped to the Go2 plant limits. Save the
   fixed tracking gains as provenance. Do not introduce gain tuning before the
   nominal executable-roll gate.
5. Compute the inverse-PD residual action from the first actually applied torque
   of each 50 Hz transition. Record unclipped and clipped actions, but inject only
   the clipped action.
6. Mirror environment bookkeeping once per control transition and use the same
   environment reward and success classifier used online.
7. Check non-foot contacts after every physics substep. Allow ordinary foot
   touchdown sequences.
8. Reject non-finite solver output, invalid initialization, non-foot contact,
   incomplete rotation, unstable landing, premature viewer closure, and fewer
   than 50 transitions.
9. Write one attempt record whether accepted or rejected; write a trajectory
   file only for accepted attempts.

CLI requirements:

- `--num-trajectories`
- `--start-seed`
- `--max-attempts`
- `--output-dir`
- `--manifest-filename`
- `--render`
- verbosity matching existing generator conventions
- an explicit commissioning-only option for nominal spread zero

Use one MPX wrapper/JIT compilation per process and re-solve per seed. File
names must include task, schema version, direction, seed, and episode length.
Write through a temporary file and atomically rename it so interrupted workers
cannot leave apparently valid `.npz` files.

Diagnostics to save and aggregate:

- solve time and solver diagnostics;
- roll progress/error over time and final orientation;
- touchdown times and contact sequence;
- minimum base height and non-foot contact count;
- MPX and total torque saturation fractions;
- unclipped residual-action range and clipping fraction;
- joint tracking error against `X`;
- success/failure reason and stability-streak length.

Gate: one nominal rendered execution and ten seeded executions pass the frozen
classifier. If optimized `X` looks correct but executed rollouts fail, stop and
tune controller execution/model alignment on this gate; do not generate data.

### Gate G4: define and validate schema v1

Proposed files:

- `mpc_rl/planner/barrel_roll_dataset.py`
- `utils/check_data_integrity.py`
- `.gitignore` for the generated dataset subtree
- `tests/test_barrel_roll_data.py`

Dataset directory:

```text
data/go2_barrel_roll/v1/
```

All values must be numeric or fixed-width Unicode so accepted files load with
`allow_pickle=False`. Required direct transition arrays:

- `policy_obs`, `next_policy_obs`: `(50, 45)`;
- `privileged_obs`, `next_privileged_obs`: `(50, 4)`;
- `actions`: `(50, 12)`;
- `rewards`: `(50,)`;
- `terminated_ctrl`, `truncated_ctrl`: `(50,)`.

Required physics/controller arrays:

- `qpos`: `(19, 201)` and `qvel`: `(18, 201)`;
- `tau_applied`, `tau_mpx`, `q_des`: `(12, 200)`;
- offline `X`, `U`, or an equally sufficient finite controller-plan record;
- phase, desired roll, measured roll progress, contacts, clipping, saturation,
  and success metrics at their documented rates.

Required scalar/provenance fields:

- integer `schema_version=1` and task ID `go2_barrel_roll`;
- robot, roll direction, rollout seed, sampled spread, and failure-free success;
- schedule, success thresholds, reward configuration, timing, action scale, LPF
  configuration, PD gains, tracking gains, and torque limits;
- explicit `domain_randomization=disabled` and `go2_sysid_enabled=false`;
- root, MPX, nested solver, and Gym-Quadruped commits;
- hashes of MPX and rollout Go2 XML files and the effective controller/task
  configuration;
- generator command/config and runtime versions.

The validator must fail nonzero on corruption, missing/wrong schema, wrong task,
wrong dimensions/timing, non-finite values, out-of-range saved actions,
inconsistent observation adjacency, incorrect recomputed phase/reward/done,
non-foot contact, incomplete roll, or unstable landing. It must report action
clipping and torque saturation but should not pretend the saved action reproduces
the direct-torque transition through the LPF.

Extend `utils/check_data_integrity.py` to accept CLI directories and delegate
task-specific validation instead of creating another generic corruption-only
tool.

Each worker gets a separate JSONL manifest. After a stage completes, create an
aggregate summary and SHA-256 checksum index for accepted files. Generated NPZ
data and raw manifests remain ignored under `/data/go2_barrel_roll/`; small
sanitized summaries may be committed outside that ignored artifact subtree,
but manifests containing machine-specific paths remain beside the dataset.

Gate: all 10 smoke files validate, intentional corruptions fail, and schema
metadata distinguishes this dataset from velocity-tracking direct files.

### Gate G5: harden direct injection and training routing

Likely files:

- `mpc_rl/common/mpc_inject_callbacks.py`
- `mpc_rl/common/quadruped_tensorboard_callback.py`
- `mpc_rl/common/__init__.py` only if a task-specific callback is needed
- `mpc_rl/train.py`
- `tests/test_barrel_roll_data.py`
- `tests/test_quadruped_model_architecture.py`

Injection work:

1. Add optional expected quadruped task/schema arguments to
   `PercentMPCInjectCallback` and validate every barrel-roll file before queuing
   transitions.
2. Sort available files before seeded selection.
3. Require direct replay for the barrel-roll task; do not fall back to the
   legacy torque-replay environment on malformed files.
4. Preserve existing distinct-transition batching across `n_envs` and `source=1`
   tagging.
5. Carry terminal success/failure and timeout information correctly into replay
   `done` and `infos`.
6. Leave legacy velocity datasets and explicit torque replay modes unchanged.

Training work:

1. Make the quadruped factory task-aware while preserving
   `velocity_tracking` as its default. Route `barrel_roll` to
   `QuadrupedBarrelRoll-v0` and reject other quadruped task strings.
2. Force robot Go2, disabled DR, disabled sysID, the barrel-roll reward, and the
   50-step horizon for this task. Fail on incompatible requested options rather
   than silently training a different task.
3. Keep SB3 asymmetric SAC and `TaggedDictReplayBuffer`. The actor remains 45D;
   SAC critic first-layer input becomes `45 + 4 + 12 = 61`.
4. Serialize the task ID, schema version, direction, schedule, reset range,
   reward config, success thresholds, timing, action/LPF/PD config, data path,
   25% target, and disabled DR/sysID settings into `config.json`.
5. Make evaluation and video recording task-aware. Do not set velocity commands
   or put velocity labels in barrel-roll video names.
6. Log roll phase/progress/error, reward components, contact/stability state,
   episode success, failure reasons, action clipping, torque saturation, and
   replay MPC percentage.
7. Evaluate on a fixed list of 100 held-out seeds that are disjoint from all
   generation and reward/controller calibration seeds. Select/checkpoint models
   by success rate, not only mean reward.

Gate: a programmatic injection test loads a validated barrel file, selects the
direct path, constructs a 45D/4D replay buffer, inserts exact saved tuples with
MPC tags across multiple vector envs, and reaches the requested percentage
without constructing a velocity replay environment.

### Gate G6: run the 100-trajectory training pilot

Generate 100 accepted trajectories only after G0-G5 pass. This gives 5,000
distinct direct transitions.

Run one SAC-MPC seed for 100,000 environment steps with production settings
except duration and dataset size:

- 25% percentage injection;
- existing quadruped SAC-MPC network and optimizer defaults initially;
- 50/200 Hz task timing;
- `action_scale=0.5` and the existing 5 Hz online LPF;
- no DR and no sysID;
- fixed held-out evaluation seeds.

Pilot checks:

- loader starts without schema/task warnings;
- injected tuples and MPC tags are correct;
- actual buffer composition tracks 25%;
- losses, Q values, entropy coefficient, actions, observations, and rewards stay
  finite;
- policy success increases above its initial level and roll-progress traces move
  in the required direction;
- controller demonstrations still pass when evaluated with the frozen
  classifier.

Do not require the final 80% production threshold at 100k. Require clear
learning progress. If there is none, follow the fallback ladder below and rerun
the pilot before scaling.

### Gate G7: generate and promote 1,000 accepted trajectories

After the 100k pilot passes, collect 1,000 accepted trajectories (50,000 direct
transitions). Run multiple terminals with disjoint, generously separated seed
ranges and unique manifest names. For example, after the CLI exists, four
workers can each target 250 successes:

```bash
conda run -n mpc-rl python mpc_rl/planner/gen_traj_data_barrel_roll.py \
  --num-trajectories=250 --start-seed=0 --max-attempts=2500 \
  --output-dir=data/go2_barrel_roll/v1_staging \
  --manifest-filename=generation_manifest_worker0.jsonl

conda run -n mpc-rl python mpc_rl/planner/gen_traj_data_barrel_roll.py \
  --num-trajectories=250 --start-seed=100000 --max-attempts=2500 \
  --output-dir=data/go2_barrel_roll/v1_staging \
  --manifest-filename=generation_manifest_worker1.jsonl
```

Workers 2 and 3 use starts `200000` and `300000`. These are proposed commands
for the planned CLI; they are not currently runnable.

Before adding workers, measure JAX compilation time, per-attempt solve time, GPU
memory, success rate, and file size from the 10- and 100-file stages. Each
process owns one MPX wrapper; do not share a compiled wrapper between processes.

After collection:

1. verify there are exactly 1,000 unique accepted seeds/files;
2. run strict validation on every file;
3. aggregate every attempt manifest and report rejection reasons;
4. create checksums and the dataset summary;
5. promote by atomic directory rename from staging to
   `data/go2_barrel_roll/v1`;
6. never add `.npz` files to Git.

Gate: the promoted directory is immutable for the production experiment. Any
action scale, reward, schema, controller, or success-threshold change creates a
new dataset version.

### Gate G8: production SAC-MPC training and evaluation

Run three 500,000-step seeds at 25% MPC replay. Add a dedicated script such as
`run_go2_barrel_roll_sac_mpc.sh` only after the final CLI is known; it should
follow existing shell-script conventions and stop on the first failed run.

A planned single-run command will have this shape:

```bash
conda run -n mpc-rl python mpc_rl/train.py \
  --env_name=quadruped-barrel_roll \
  --robot=go2 \
  --algorithm=SAC-MPC \
  --total_timesteps=500000 \
  --inject_type=percentage \
  --percentage=25 \
  --quadruped_mpc_replay_mode=direct \
  --data_dir=data/go2_barrel_roll/v1 \
  --domain_rand=False \
  --domain_rand_config_type=disabled \
  --use_go2_sysid=False \
  --seed=<seed> \
  --suffix=go2-barrel-roll-v1
```

This is also a planned command, not currently runnable. Confirm final flag
spelling against the implemented Abseil CLI.

For each seed:

- preserve all checkpoints and success-rate evaluations;
- record the exact dataset checksum index and source revisions;
- evaluate the selected checkpoint on the same 100 held-out initialization
  seeds;
- save representative success and failure videos;
- report success rate, failure-reason distribution, roll-progress error,
  touchdown/stabilization timing, and return.

MVP experiment acceptance: median success across the three policies is at least
80%. Report every seed and the aggregation rule. Do not tune on the 100 held-out
seeds or replace a weak seed without reporting it.

## Test and validation matrix

| Layer | Required evidence | Failure response |
| --- | --- | --- |
| Existing velocity task | No new failures beyond recorded baseline | Fix regression before continuing |
| MPX config/reference | Go2 IDs/values, finite one-turn 1 s plan | Fix config/reference |
| Roll math | Correct quaternion continuity and signed unwrapping | Fix shared task math |
| Reset | Seeded `[0, 0.10]` stance spread and valid contacts | Fix sampler |
| Environment | 45D/4D spaces, 50/200 Hz, correct reward/done/info | Fix environment |
| Executed controller | Ten successful closed-loop task-plant rollouts | Tune execution/model; do not generate |
| Schema | Strict v1 validation and corruption rejection | Fix generator/validator |
| Injection | Exact direct tuples, MPC tags, multi-env batching, 25% | Fix loader/callback |
| Model | SAC actor 45D, critic input 61, finite update | Fix routing/policy config |
| Pilot | Increasing held-out success by 100k | Use fallback ladder |
| Production data | 1,000 unique valid files and checksums | Regenerate failed shards |
| Production policy | Median >=80% over three x 100 held-out episodes | MVP result not achieved; report and diagnose |

Suggested focused test command after implementation:

```bash
conda run -n mpc-rl pytest -q \
  tests/test_barrel_roll_env.py \
  tests/test_barrel_roll_data.py \
  tests/test_velocity_tracking_env.py \
  tests/test_quadruped_model_architecture.py
```

Also run the repository's broader relevant suite before declaring the code
complete. Do not weaken the three known unrelated stale tests to make this
feature appear green; report baseline and post-change results explicitly.

## Failure diagnosis and fallback order

Use this order so failures do not trigger unrelated redesigns:

1. **Optimized plan invalid:** correct Go2 config, contact/reference indexing, or
   MPX solver setup.
2. **Plan valid but task-plant execution fails:** inspect MPX/Gym model
   differences, node-rate mapping, tracking error, feedback gains, torque
   saturation, and contact timing. Tune only on commissioning seeds.
3. **Execution succeeds but saved actions clip heavily:** increase
   `action_scale`, regenerate a new schema/dataset version, and rerun the
   100-trajectory pilot.
4. **Injection incorrect:** fix strict schema routing and replay batching before
   changing learning configuration.
5. **Critic losses/value estimates fail around touchdown:** expand privileged
   observations from 4D to the documented 9D fallback and create a new schema
   version.
6. **Learning is stable but the policy does not rotate:** inspect phase features,
   unwrapped tracking/progress reward, normalization, and demonstration coverage.
7. **Policy rotates but does not land:** inspect success bonus, terminal handling,
   landing demonstrations, and late-phase state coverage before adding shaping.
8. **Only after the above:** tune SAC hyperparameters or add a narrowly justified
   reward term. TD3-MPC, torque actions, LPF removal, DR, and sysID remain outside
   the initial fallback ladder.

Every fallback that changes observations, action scale, reward semantics,
controller behavior, or success thresholds invalidates the existing dataset
version and must be reflected in config serialization and documentation.

## Documentation and commit sequence

Update after the corresponding behavior exists:

- `README.md`: task name, generator/validator commands, data location, training
  and evaluation commands, 50/200 Hz timing, no-DR/no-sysID scope, and artifact
  outputs;
- this plan: mark gates complete with exact validation results and deviations;
- `docs/HZ_CONTROL_REFERENCE.md`: add the barrel-roll row only if the new task
  changes or clarifies user-visible timing semantics;
- `docs/low_pass_filter_implementation.md`: state explicitly that online
  barrel-roll actions use the LPF while MPC demonstration generation follows
  the existing direct-torque/inverse-PD pipeline;
- MPX README/example comments: Go2 barrel-roll invocation and finite smoke use;
- the new experiment script: exact dataset, schema, percentage, disabled
  DR/sysID, seeds, and budgets.

Recommended commit order:

1. MPX Go2 barrel-roll cleanup, diagnostics, tests, and example;
2. root shared task semantics, environment, and environment tests;
3. generator, schema validator, manifest/checksum workflow, and tests;
4. strict injection, training/evaluation routing, logging, and tests;
5. documentation and pilot/run scripts;
6. root MPX gitlink update if it was not already included deliberately.

Keep root and submodule histories reviewable. Do not combine generated data,
training logs, or exported policies with source commits.

## Explicit non-goals

- pure SAC or other 0% comparison training for the initial MVP;
- TD3-MPC;
- left/right command-conditioned or repeated barrel rolls;
- random terrain, DR, pushes, observation noise, encoder bias, or sysID;
- torque-action policies;
- imitation/behavior-cloning losses;
- redesigning the replay-buffer sampling algorithm;
- repairing unrelated velocity-task audit findings or stale tests;
- deployment, ONNX export, or real-robot execution;
- broad refactoring of `train.py` or the velocity environment;
- generating more than the gated dataset size before the prior gate passes.

## Final implementation handoff checklist

Before claiming completion, report:

- source files and submodule commits changed;
- why each public behavior/config/schema change was necessary;
- exact test commands and pass/fail/skip counts, including known baseline gaps;
- nominal and varied controller rollout evidence;
- dataset counts, rejection reasons, schema version, checksum summary, storage,
  and generation throughput;
- injection composition and tuple-parity evidence;
- pilot and three-seed production training configurations;
- all three 100-episode held-out success rates and their median;
- remaining risks, failed gates, fallbacks used, and deviations from this plan.
