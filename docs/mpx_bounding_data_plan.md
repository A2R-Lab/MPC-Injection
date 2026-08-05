# MPX Bounding-Data Implementation Plan

## Goal and definition of done

Build a reproducible pipeline that uses the checked-in `deps/mpx` controller to
produce Unitree Go2 **bounding** demonstrations that are valid inputs to the
current SAC-MPC and TD3-MPC replay-buffer path.

For this plan, a symmetric bound means:

- `FL` and `FR` are phase-aligned;
- `RL` and `RR` are phase-aligned;
- the front pair and rear pair are approximately half a cycle apart;
- production data shows the pattern in measured contacts, not only in the MPX
  reference schedule;
- if an aerial phase is required, it must be observed and measured. Merely
  changing phase offsets at `duty_factor=0.5` creates paired front/rear support
  but does not by itself prove a flight phase.

The work is complete only when all of the following hold:

1. A clean root branch and a coordinated MPX submodule branch contain the
   implementation.
2. Gait parameters are explicit per controller instance and survive controller
   reset; the trot defaults remain unchanged.
3. Replaying saved actions through the actual training environment reproduces
   the saved transitions within a declared numerical tolerance.
4. A validator classifies accepted files as bounds from measured foot contacts
   and rejects falls, non-foot ground contact, invalid shapes, non-finite data,
   excessive clipping/saturation, metadata mismatch, and transition mismatch.
5. A small nominal pilot and a small domain-randomized pilot pass before bulk
   generation.
6. SAC-MPC or TD3-MPC can inject the files through the direct-transition path,
   and a trained policy retains the bounding contact pattern during evaluation.
7. The exact root commit, MPX commit, controller configuration, environment
   configuration, seeds, and dataset checksums are recorded.

Do not start a 10,000-trajectory collection before the action/transition parity
and gait-classification gates pass.

## What the repository currently does

The relevant path is:

```text
MPX gait timer and reference generator
  -> MPCControllerWrapper.run(qpos, qvel, command, measured_contact)
  -> (feedforward torque, desired joint position, desired joint velocity)
  -> QuadrupedVelocityTrackingEnv rollout
  -> compressed .npz direct transitions plus diagnostic torque/state arrays
  -> PercentMPCInjectCallback
  -> TaggedDictReplayBuffer (source=1 for MPC)
  -> SB3 SAC-MPC or TD3-MPC training
```

Key files and behavior:

- `deps/mpx/mpx/config/config_go2.py` defines the Go2 model, 50 Hz MPC,
  25-stage horizon at `dt=0.02`, gait phase timers, duty factor, step frequency,
  step height, costs, and torque bounds.
- `deps/mpx/mpx/utils/mpc_utils.py::timer_run` advances each normalized leg
  timer by `dt * step_freq`, wraps it into `[0, 1]`, and declares stance while
  `timer < duty_factor`.
- MPX contact order is explicitly `[FL, FR, RL, RR]` in `config_go2.py` and
  `models.py`.
- The current Go2 timer is `[0.5, 0.0, 0.0, 0.5]`. At the first 50 Hz update
  with `step_freq=2.0` and `duty_factor=0.5`, this produces diagonal support
  `[0, 1, 1, 0]`, i.e. a trot reference.
- A bound phase seed is `[0.5, 0.5, 0.0, 0.0]` (rear pair initially in stance)
  or the phase-equivalent `[0.0, 0.0, 0.5, 0.5]` (front pair initially in
  stance). Startup behavior must determine which orientation is more stable;
  steady-state gait identity is the same.
- `mpc_rl/planner/gen_traj_data_mpx_dr.py` is the current canonical generator.
  It can run with a named startup-DR preset, saves direct RL transitions, saves
  the realized DR patch, rejects incomplete/fallen trajectories, and writes an
  attempt manifest.
- Passing `--domain-rand-config-type disabled` through that generator already
  covers nominal generation. `gen_traj_data_mpx.py` is a legacy torque/state
  generator and should not become a second independent bounding implementation.
- The generated direct schema currently includes policy/privileged observations,
  actions, rewards, terminations, commands, MuJoCo states, MPX and applied
  torques, desired positions, timing, DR state, and sysID metadata.
- `PercentMPCInjectCallback` detects that direct schema, batches unique
  transitions across vector-environment slots, tags them as MPC, and injects
  until the requested buffer percentage is reached.
- Quadruped MPC algorithms use the environment's simplified reward. It tracks
  forward velocity and termination, but does not explicitly reward a bound.
  The full reward is not used by SAC-MPC/TD3-MPC and, if enabled, currently
  rewards diagonal trot timing and penalizes non-diagonal two-foot support.
- The checked-in data directories described by `README.md` are not present in
  this worktree. They are ignored/generated artifacts, so a new dataset needs
  an explicit storage and provenance decision.

## Verified baseline and constraints

The investigation was performed at:

- root branch `low-pass-filter`, commit
  `625fe20fc7fb4276d891e6a0456b6c983c88fa64`;
- MPX branch `roy/mpx-local-changes`, commit
  `df82900a5283f58878efe5118095e3136d9ffdd6`;
- nested `primal_dual_ilqr` commit
  `273d78dec439ed270af6f9cb8340af801fefd332`.

At investigation time, the root worktree was dirty with unrelated user changes
and the MPX submodule had an uncommitted `config_go2.py` change increasing the
roll cost in `Qrot` from 5,000 to 10,000. Those changes were subsequently
checkpointed at root commit
`55fe2464bb2c0389829100a29904fb845e89ce99` and MPX commit
`249c7323ab0cd13bea2ddb8ed5f252eb9ccde85c`. The checkpoint preserves the
edit; it does not approve the 10,000 value for the bounding experiment. That
choice must still be explicit in controller configuration and dataset metadata.
Untracked exported policy artifacts remain outside the WIP commit.

The `mpc-rl` Conda environment matches `environment.yml` for the important
runtime packages: Python 3.11, JAX/JAXLIB 0.6.2, MuJoCo/MJX 3.3.6,
gym-quadruped 1.1.1, MPX 0.1, Stable-Baselines3 2.7.0, and SBX 0.23.0. The
repository-local `.venv` does not match and is not suitable for this work.

Six existing targeted generator/replay/injection tests pass in the Conda
environment. They prove the current short-horizon schema, DR replay, manifest,
and direct injection plumbing. They do **not** prove a bound, actual MPX solver
quality, or action/transition consistency through `env.step()`.

The MPX prediction XML and gym-quadruped rollout XML are not identical. They
differ in contact solver settings, collision geometry, joint dynamics defaults,
and other details. The local sysID hook patches joint dynamics in the MPX model,
but a sampled rollout DR patch is not copied into the already-compiled MJX
prediction model. This is a nominal-model controller operating on a randomized
plant, not an oracle MPC with the same randomized model. Preserve and record
that behavior unless a separate experiment intentionally changes it.

## Transition-parity gate: measure before changing the rollout

The current DR generator is operational: it generates files that the existing
replay-injection path can load and use. It derives a clipped inverse-PD action
from the first simulation substep, then advances MuJoCo by applying MPX
feedforward plus hard-coded `Kp=10`, `Kd=2` torques directly. The training
environment instead:

- uses per-joint gains (`Kp` 20/20/40 and `Kd` 1/1/2 before DR);
- low-pass filters absolute joint targets at 5 Hz by default;
- clips the action to `[-1, 1]`;
- clips torques to the realized actuator limits; and
- holds one filtered joint target across four simulation substeps.

Therefore, transition equivalence is not established merely because generation
and injection run successfully. A one-step diagnostic with the existing fake
MPC showed a maximum qpos difference of about `8e-4` and a policy-observation
difference of about `5.8e-2` after calling `env.step(saved_action)`. This result
does not establish that the mismatch is material for real MPX trajectories or
training, and it is not sufficient reason by itself to replace a working
rollout path.

Treat parity as an empirical decision gate:

1. Before changing generation, define the compared fields, numerical metrics,
   and acceptance tolerances. Include qpos, qvel, policy and privileged
   observations, reward, termination, and multi-step drift. Base tolerances on
   deterministic numerical repeatability and the precision required by the
   direct-transition loader, not on thresholds chosen after seeing the result.
2. Replay saved actions from saved initial state through the actual training
   environment for one-step and full short rollouts. Run the check with actual
   MPX output under nominal dynamics and at least one nontrivial deterministic
   DR patch, including the action LPF, realized gains, torque limits, clipping,
   and deterministic pushes.
3. Report maximum and RMS disagreement per field, action-clipping and
   torque-saturation rates, and how disagreement accumulates over the rollout.
4. If the current inferred-action pipeline passes the declared tolerances,
   preserve it. Record the action-conversion method and parity report in the
   schema and provenance; do not reimplement the rollout solely for theoretical
   exactness.
5. If it fails, make the training environment's action path the source of truth
   for files advertised as accepted direct transitions. Start by mapping MPX
   output to `q_target_desired = q_des + tau_ff / kp_realized`, invert the
   first-order target filter, convert and clip the raw target exactly as
   `env.step()` does, and save the transition returned by `env.step(action)`.
   Treat this mapping as a candidate to validate, not as an assumed identity.
6. If no action mapping produces stable rollouts and acceptable parity without
   excessive clipping or saturation, stop and decide on an environment
   action-interface change. The direct-torque files can remain useful
   diagnostics, but must be labeled as non-direct rather than silently admitted
   to the accepted direct-transition dataset.

## Implementation plan

Execute this as gated work, not as thirteen independent tasks:

| Gate | Evidence required to pass | Work unlocked |
| --- | --- | --- |
| G0: isolated baseline | Clean feature worktree, exact submodule commits, baseline tests, runtime/model hashes | MPX API changes |
| G1: planned gait | MPX phase/reset tests and finite nominal solver smoke run | Environment/action adapter |
| G2: validated transition | Predeclared parity tolerances; one-step and full-short-rollout replay report under nominal and nontrivial DR; conditional action-path remediation only if the baseline fails; substep contact diagnostics | Gait calibration |
| G3: nominal bound | Frozen classifier/config passes a disjoint nominal acceptance set | DR pilot |
| G4: robust bound | DR acceptance set passes with quality reported by DR/command bin | Injection/training pilot |
| G5: learned bound | Exact direct tuples are injected and policy rollouts pass the bound classifier across the planned seed/speed matrix | Bulk generation |
| G6: released dataset | Every file validates; checksum index, provenance, and aggregate report are frozen | Production training |

Each gate produces a small machine-readable report. A failed gate returns to
the immediately relevant phase; it does not justify weakening acceptance rules
or starting the next phase.

### 1. Isolate the work before changing code

Use a separate worktree so the current dirty `low-pass-filter` worktree remains
untouched. Do not stash or reset the user's changes.

```bash
git worktree add ../MPC-RL-mpx-bound \
  -b feature/mpx-bound-data \
  625fe20fc7fb4276d891e6a0456b6c983c88fa64
cd ../MPC-RL-mpx-bound
git submodule update --init --recursive
git -C deps/mpx switch -c feature/bound-gait-api
```

The plan file in the original dirty worktree is untracked and will not
automatically appear in a worktree created from the commit above. Bring over
only this reviewed document (or cherry-pick a docs-only commit) after creating
the feature worktree; do not copy the dirty worktree wholesale. Before running
the commands, confirm that neither proposed branch name nor target worktree
path already exists and choose a non-conflicting equivalent if needed.

Before implementation, decide explicitly whether MPX checkpoint commit
`249c7323ab0cd13bea2ddb8ed5f252eb9ccde85c`, which changes
`config_go2.py::Qrot`, belongs in the experiment. If it does, bring that commit
onto the MPX feature branch and identify it in metadata. If it does not, leave
the clean branch at the baseline 5,000 value. Do not copy the entire WIP root
commit into the feature branch.

Because `deps/mpx` is a Git submodule with its own `origin` fork, use two
coordinated commits:

1. commit and push the MPX API change on `feature/bound-gait-api`;
2. commit the superproject generator/tests/docs plus the updated MPX gitlink on
   `feature/mpx-bound-data`.

The nested `primal_dual_ilqr` submodule should remain unchanged.

The expected implementation surface is deliberately narrow:

- `deps/mpx/mpx/utils/mpc_wrapper.py` and one small gait-preset module under
  `deps/mpx/mpx/config/`, plus MPX-local tests;
- `mpc_rl/envs/velocity_tracking_env.py` for exact control-step diagnostics,
  without creating a second simulation loop;
- `mpc_rl/planner/gen_traj_data_mpx_dr.py` for canonical collection;
- `mpc_rl/common/mpc_inject_callbacks.py` and `mpc_rl/train.py` for strict
  loading/configuration;
- `utils/check_data_integrity.py` plus focused tests under the existing
  `tests/` tree;
- `README.md`, the MPX README, and one bound experiment script/config.

These are proposed edit locations, not a requirement to touch every file. Reuse
an existing test module when its scope remains clear.

### 2. Freeze a reproducible baseline

From the clean worktree:

```bash
conda run --no-capture-output -n mpc-rl \
  pytest -q tests/test_velocity_tracking_env.py \
  -k 'disabled_dr_generator or dr_replay_matches or nominal_replay_path or generation_smoke or saved_transition_parity or direct_transition_injection'

conda run --no-capture-output -n mpc-rl \
  python deps/mpx/mpx/examples/mjx_quad.py
```

The interactive MPX example is a manual smoke check; close it after confirming
the committed trot behavior. Record the GPU model, CUDA driver, Conda package
versions, root/MPX/nested-submodule commits, and hashes of both Go2 XML files.
JAX compilation and performance are hardware-sensitive.

### 3. Add an instance-owned gait API to MPX

Modify `deps/mpx/mpx/utils/mpc_wrapper.py` without changing the existing return
signature of `MPCControllerWrapper.run()`.

Required behavior:

- A controller owns its initial phase vector, moving duty factor, step
  frequency, and step height. Do not mutate `mpx.config.config_go2` globals.
- Validate phase shape against `config.n_contact`, finite values, normalized
  range, `0 < duty_factor <= 1`, positive frequency, and non-negative height.
- `reset()` restores the controller's selected initial phase, not always
  `config.timer_t`.
- Expose an opt-in diagnostic containing the planned contact schedule used by
  the reference generator. Do not infer it later from a separately advanced
  timer; the wrapper currently advances the timer before the reference
  generator advances its horizon.
- Preserve current constructor defaults and trot behavior when no gait override
  is provided.
- Either give `BatchedMPCControllerWrapper` the same instance/data semantics or
  explicitly reject unsupported gait overrides there. Do not leave two APIs
  that silently interpret the same parameter differently.

Add named Go2 gait presets in one place, preferably a small new MPX config
module rather than another copy of the whole Go2 controller config:

```text
trot:
    phase = [0.5, 0.0, 0.0, 0.5]

bound_hind_first:
    phase = [0.5, 0.5, 0.0, 0.0]

bound_front_first:
    phase = [0.0, 0.0, 0.5, 0.5]
```

Keep `duty_factor`, frequency, height, and any cost override explicit beside the
selected dataset configuration. Do not hard-code a production bound duty factor
yet. At 0.5 the paired support can be commissioned without flight; duty factors
below 0.5 can introduce flight windows and must be tuned and validated.

MPX tests should prove:

- exact phase-to-contact mapping in `[FL, FR, RL, RR]` order;
- reset persistence;
- invalid parameter rejection;
- unchanged default trot output;
- front-pair and rear-pair planned-contact synchronization for both bound phase
  orientations;
- a below-0.5 duty factor creates a planned all-flight interval over a full
  cycle.

### 4. Make the DR generator canonical and gait-aware

Extend `mpc_rl/planner/gen_traj_data_mpx_dr.py`. Use it for both nominal and DR
data; make `gen_traj_data_mpx.py` a documented legacy path or a thin wrapper
rather than duplicating new gait logic.

Add CLI/config inputs for:

- `--gait` with at least `trot`, `bound_hind_first`, and
  `bound_front_first`;
- optional `--duty-factor`, `--step-frequency-hz`, and `--step-height-m`
  overrides;
- explicit command bounds for `vx`, `vy`, and `wz`;
- command ramp duration and optional standing warmup duration;
- dataset schema version and output dataset name;
- an explicit action-conversion mode that distinguishes the existing
  inferred-action/direct-torque rollout from any environment-step mode added
  after a failed parity gate;
- strict clipping/saturation rejection limits;
- optional deterministic seed-shard identifiers for parallel generation.

Replace `_configure_mpc_duty_factor()`'s hard-coded moving value of `0.5` with
the selected gait's moving duty factor. Standing can continue to use
`duty_factor=1.0`; returning to motion must restore the selected value.

The applied command—not merely the final sampled command—must be stored during
warmup and ramps. Keep initial-state and command RNG streams separate from the
DR RNG stream. Seed ranges for parallel jobs must be disjoint.

For each control step, save:

- measured foot contacts in fixed `[FL, FR, RL, RR]` order;
- the MPX planned contact vector/schedule used for that solve;
- raw and clipped action;
- desired and actually filtered joint targets;
- action clipping mask/magnitude and torque saturation mask/magnitude;
- achieved body-frame velocity and base roll/pitch/height;
- existing observations, rewards, terminations, qpos/qvel, MPX diagnostics,
  commands, and realized DR state.

Do not weaken the existing rejection of falls, incomplete episodes, torso
ground contact, or other non-foot robot-ground contact. The current generator
checks non-foot contact after each of four direct-torque simulation substeps,
whereas `env.step()` exposes only the completed control step. Preserve that
coverage by adding opt-in per-substep diagnostics/observation inside the
environment's existing decimation loop, or by factoring the exact environment
control loop into one shared helper. Do not reintroduce a duplicate rollout
loop merely to inspect contacts. The diagnostics should accumulate applied
torques, contacts, and non-foot contact across all four substeps and leave
normal training behavior unchanged when disabled.

### 5. Version and validate the dataset schema

Add scalar/array metadata that can be read without arbitrary Python objects:

- `schema_version`;
- `gait_name`, `contact_order`, phase offsets, duty factor, frequency, and
  height;
- command bounds, warmup, ramp, and resample settings;
- action-conversion mode, action scale, LPF cutoff/alpha, realized Kp/Kd, and
  torque limits;
- reward mode and environment timing;
- DR preset and realized patch;
- Go2 sysID toggle and signature;
- root, MPX, and primal-dual-iLQR commit IDs;
- hashes of MPX and rollout Go2 XML files;
- generator invocation or normalized config, including a deterministic hash of
  all effective controller weights/bounds so an uncommitted numeric tuning
  cannot hide behind a commit ID;
- seed, DR seed, shard ID, generation timestamp, and host/GPU/runtime versions.

Use a filename that prevents gait mixing, for example:

```text
quadruped_dr_<dr-preset>_gait_bound_hind_first_seed_<seed>_ep_<steps>.npz
```

Extend the JSONL manifest so every attempted seed records success/failure,
failure reason, gait configuration, summary gait metrics, clipping/saturation,
tracking error, and file checksum when saved. Write a dataset-level summary and
checksum index only after validation.

Extend `utils/check_data_integrity.py` instead of creating a second generic
corruption checker. A gait-specific validator module can provide the schema and
physics checks and be called by that utility and tests. Convert the utility's
hard-coded directory list into CLI path arguments and a nonzero exit status on
any invalid file. New-schema loaders should use `allow_pickle=False`; if legacy
object-bearing files must remain loadable, isolate that compatibility path and
never enable it for an accepted bound dataset.

Strict per-file checks:

- all required direct-transition and metadata keys exist;
- all shapes agree with `episode_length`, 50 Hz control, 200 Hz simulation,
  12 joints, 45 policy observations, and 3 privileged observations;
- every numeric array is finite;
- actions are in range and saved rewards/terminations match recomputation;
- saved commands match command slots in policy observations;
- a full saved-action replay from the saved initial condition matches saved
  qpos/qvel and observations within declared tolerances;
- per-substep diagnostics prove the non-foot contact check covered the entire
  transition, not only the final simulation state;
- no fall or non-foot ground contact occurs in accepted data;
- controller/environment/sysID/DR/gait metadata is internally consistent;
- file checksum matches the index.

Measured bound metrics should include:

- front-pair agreement `mean(FL == FR)`;
- rear-pair agreement `mean(RL == RR)`;
- front/rear phase separation from contact transition timing;
- fractions of front-only, rear-only, all-four, diagonal-two, lateral-two, and
  all-flight support states while a movement command is active;
- observed stride frequency and left/right touchdown-time differences;
- velocity tracking error by command bin;
- roll/pitch extrema, minimum base height, torque saturation fraction, action
  clipping fraction, and non-foot contact count.

Separate data used to tune/classify the gait from data used to accept it.
Derive numerical gait-quality thresholds from a deterministic calibration set,
freeze them in versioned validator configuration, and then evaluate disjoint
nominal and DR seed ranges. Keep invariant correctness rules—finite arrays,
shape/schema consistency, action replay parity, no non-foot contact, and no
falls—strict rather than data-derived. Do not choose thresholds after inspecting
an acceptance set or the bulk dataset.

### 6. Resolve controller tuning with small deterministic trials

First change only the phase preset and retain the committed MPX controller
parameters. Run deterministic, rendered nominal trials at the lower, middle,
and upper parts of the current training command range (`vx` is currently
`[0.0, 0.5]`; `vy=wz=0`). This isolates gait scheduling from cost tuning.
Treat `vx=0` as a separate stand/warmup check, not as evidence for a locomotion
gait classifier. A concrete initial calibration matrix is both startup phase
orientations at `vx={0.1, 0.3, 0.5}` m/s: six rendered trajectories, followed
by repeated unrendered seeds only where behavior is stable. These values are a
commissioning proposal within the verified current range, not claimed
controller limits.

Commission in this order:

1. paired front/rear schedule at `duty_factor=0.5`;
2. both startup phase orientations;
3. modest reductions below 0.5 to introduce a measured flight interval;
4. step frequency and height;
5. command warmup/ramp behavior;
6. only then MPX cost weights or robot-height changes.

The current MPX Go2 config strongly penalizes pitch and angular velocity. A
true bound naturally has sagittal pitch and vertical oscillation, so those
weights may suppress it. If tuning is required, create an explicit bound
controller preset; do not overwrite the trot defaults or rely on an uncommitted
`config_go2.py` edit. Change one parameter group per trial and record every
trial in a machine-readable calibration table.

Do not expand the command range beyond the training environment's current
range merely to make bounding easier. If a stable bound requires higher speed,
treat the command range as a coordinated environment/training interface change:
parameterize it in `train.py`, serialize it into `config.json`, generate data
from the identical range, and add evaluation commands in that range.

### 7. Measure and prove direct-transition parity

Before accepting any pilot file, define parity tolerances and test the current
rollout without assuming it must fail:

- one saved action replayed through the real LPF/gain/clip path reproduces the
  saved one-step transition within the declared tolerance;
- a full short trajectory replaying only saved actions reproduces qpos, qvel,
  policy observations, privileged observations, reward, and termination;
- the same checks pass with a nontrivial DR patch, realized Kp/Kd, motor
  strength, and deterministic pushes;
- transient non-foot contact on an intermediate simulation substep is rejected;
- clipped actions remain honest transitions and are counted for rejection;
- disabling DR still uses the nominal sysID/contact-friction plant;
- legacy torque-only files cannot be mistaken for the new direct schema.

If the existing inferred-action rollout passes, keep it and store its conversion
mode, declared tolerances, and measured parity summary. If it fails, implement
and validate the environment-step mapping described in the transition-parity
gate, then rerun the same tests. Keep torque arrays for controller analysis, but
the direct-injection callback must consume only transitions that carry the new
schema version and passed action-replay parity. A failing baseline is evidence
to remediate; it is not permission to weaken the tolerances.

### 8. Generate and accept a nominal pilot

Generate a small pilot in a new, gait-specific directory. Start with a handful
of fixed-command rendered trajectories for visual inspection, then tens of
seeded non-rendered trajectories across command bins. Do not mix calibration
files with accepted files or reuse their seed ranges. After the six-run
calibration matrix in phase 6, an initial acceptance proposal is five new seeds
for each stable phase-orientation/speed combination (up to 30 trajectories).
This is a commissioning sample, not statistical evidence for a production
success rate; increase it before G3 if metrics sit near a frozen threshold.

Example shape of the command after the new CLI exists:

```bash
conda run --no-capture-output -n mpc-rl \
  python mpc_rl/planner/gen_traj_data_mpx_dr.py \
  --gait=bound_hind_first \
  --domain-rand-config-type=disabled \
  --num-trajectories=<pilot-count> \
  --episode-length=1000 \
  --start-seed=<pilot-seed> \
  --output-dir=data/quadruped_bound/nominal_pilot
```

The exact new flag spelling is part of the implementation and must be reflected
in `--help` and `README.md` before use.

Accept the nominal pilot only if:

- action replay parity passes every file;
- the measured contact classifier identifies the requested bound;
- the desired flight requirement, if enabled, is observed;
- all files complete without falls or non-foot contact;
- tracking, clipping, torque saturation, posture, and solver timing are within
  the thresholds frozen on the separate calibration set;
- repeated generation with the same seed produces identical transition arrays
  and metadata apart from allowed timestamps/timing diagnostics.

Estimate storage and throughput from this pilot. A 1,000-control-step file
contains multiple 200 Hz float64 state/torque arrays plus direct observations;
do not assume compressed size. Measure mean/p95 file size and accepted
trajectories per GPU-hour before setting a bulk count.

### 9. Generate and accept a DR pilot

Use the same gait/controller/action conversion with the training DR preset. The
current experiment script uses `sysid_dyn20_mjlab`; that should be the initial
DR target unless a different training preset is intentionally selected.

Generate a small stratified sample first. As an initial commissioning matrix,
use ten new seeds per stable speed/orientation combination (up to 60
trajectories), with seed ranges disjoint from calibration and nominal
acceptance. This does not establish tail robustness; expand the sample or use a
designed fixed patch matrix when coverage of a DR parameter is poor. Confirm
that:

- the generator and training run use the same DR preset and sysID toggle;
- clean direct observations versus noisy training observations are an explicit
  choice. The current generator zeros observation noise and encoder bias for
  demonstrations even when dynamics DR is enabled;
- the nominal MPX prediction model versus randomized rollout plant behavior is
  recorded;
- gait metrics and success rate remain acceptable across realized friction,
  mass/CoM, joint dynamics, gains, motor strength, and push conditions;
- action replay parity includes the deterministic push schedule.

If the DR success rate collapses, tune the controller on a fixed, versioned
calibration matrix. Do not silently reject most of the DR range until only easy
plants remain; report acceptance by DR parameter bins so dataset selection bias
is visible.

### 10. Harden injection and training configuration

Update `mpc_rl/common/mpc_inject_callbacks.py` and `mpc_rl/train.py`:

- add an expected quadruped dataset gait and schema version to training config;
- sort trajectory filenames before seeded selection;
- validate each loaded file's schema, gait, timing, observation/action shapes,
  sysID, and DR preset;
- load accepted direct-schema files with `allow_pickle=False`;
- fail on schema/gait/timing/shape mismatch rather than merely warning;
- preserve direct-transition batching into `TaggedDictReplayBuffer`;
- retain legacy torque replay only behind its explicit modes;
- serialize dataset path, expected gait/schema, command range, action LPF,
  reward mode, DR preset, and sysID toggle into `config.json`.

Use `inject_type=percentage` for this path. The current percentage callback is
the proven quadruped direct-transition implementation. Do not assume the older
fixed callback has equivalent quadruped schema/DR behavior without separate
tests.

The current simplified reward does not identify a gait. Bounding demonstrations
may bias learning, but velocity reward alone does not guarantee the trained
policy will retain a bound. Use the measured gait classifier on policy rollouts
as a required acceptance test. If the policy converges to another gait, stop
and make the objective explicit rather than generating more of the same data:

- add a bound-aware contact schedule/reward selected by an explicit gait mode,
  or add a separately justified imitation objective;
- keep the trot defaults unchanged;
- if the full reward is used, replace/disable its diagonal-trot reward and its
  `bad_two_foot_contacts` penalty for bound mode;
- decide whether policy observations need an explicit gait phase/clock. Any new
  observation must also be available in deployment and changes the saved data
  schema/model interface.

This objective change is conditional on the policy-retention experiment; it is
not required merely to make valid bounding data.

### 11. Run an end-to-end training pilot

First run a programmatic callback test that loads only accepted bound files and
proves:

- the direct path is selected;
- no legacy torque replay environment is constructed;
- replay-buffer observations/actions/rewards equal the saved transitions;
- all inserted tags are MPC and the reported percentage is correct with more
  than one vector environment;
- deterministic file selection is stable.

Then run a short SAC-MPC or TD3-MPC training pilot with the same settings used
for production except duration and dataset size. Use a separate log directory
and include `gait-bound` in the suffix. Confirm:

- startup injection completes without loader warnings;
- gradients run and losses remain finite;
- actual replay-buffer MPC percentage tracks the target;
- evaluation does not immediately fall;
- policy rollouts, not demonstration rollouts, pass the bound classifier at
  multiple commanded speeds.

Compare at least:

- 0% MPC control;
- the intended bound MPC percentage (the current primary scripts use 25%);
- three training seeds for each condition as the initial matrix.

Do not compare runs with different action filters, rewards, DR presets, sysID
settings, command ranges, network architecture, or dataset schema as if MPC
percentage were the only variable.

### 12. Scale bulk generation safely

After both pilots and training retention pass:

- choose the bulk count from measured training need, disk budget, and pilot
  throughput rather than copying the old `10k` directory name;
- split work into disjoint seed ranges and separate shard manifests;
- use one compiled MPX controller per worker and confirm GPU memory before
  increasing worker count;
- write to staging shard directories, then validate and promote accepted files
  into an immutable dataset directory;
- never let two workers write the same filename or manifest;
- preserve rejected-attempt manifests;
- run corruption, schema, checksum, action-parity, and gait-quality validation
  over every accepted file;
- aggregate success and quality by command bin and DR parameter bin;
- mark the dataset complete with a frozen summary, checksum index, and exact
  generation command.

The `.npz` files are gitignored and should not be committed. Commit the dataset
manifest/summary only if it is small and contains no machine-specific sensitive
paths; otherwise store it beside the dataset in the chosen artifact store and
link it from experiment documentation.

Before G6, choose and record the actual artifact-store path, retention policy,
and available disk budget. The repository does not currently specify them, so
this plan does not invent a production destination.

### 13. Documentation and commit sequence

Update at least:

- `README.md`: canonical nominal/DR bound generation, validation, and training
  commands; schema and artifact location;
- MPX `README.md`: instance gait override and contact-order semantics;
- the relevant quadruped experiment script or a new bound-specific script:
  explicit data directory, gait/schema expectation, DR/sysID, replay mode, and
  command range;
- tests/docstrings that still describe direct torque application as a valid
  direct RL transition.

Recommended commit order:

1. MPX gait API and MPX tests;
2. root transition-parity tests and report, plus an action-semantics fix only
   if the baseline fails;
3. gait-aware generator, schema, diagnostics, and validators;
4. strict loader/training config and injection tests;
5. docs and pilot configuration;
6. MPX submodule gitlink update in the root commit series.

## Validation matrix

| Layer | Required check | Failure means |
| --- | --- | --- |
| MPX timer | Bound phase produces paired planned contacts in correct order | Gait preset/API is wrong |
| MPX reset | Selected gait survives every trajectory reset | Shared-controller generation is wrong |
| Solver | Finite solution, acceptable solve time, no NaN fallback | Controller/config is not usable |
| Action semantics | Saved-action replay matches the saved transition within predeclared tolerances | Preserve the report if it passes; remediate the action path if it fails |
| Gait | Measured contacts meet frozen bound classifier | Reference did not create a bound |
| Safety | No fall, non-foot contact, or invalid posture | Trajectory is rejected |
| Actuation | Clipping and saturation within frozen bounds | Action representation/controller needs tuning |
| Tracking | Achieved velocity acceptable in each command bin | Demonstration quality is insufficient |
| DR | Quality remains acceptable across DR bins | Dataset is biased or controller is not robust |
| Schema | Shapes, versions, provenance, hashes, checksums match | File is rejected |
| Injection | Direct path inserts exact saved tuples and tags | Training plumbing is wrong |
| Training | Finite learning and requested MPC percentage | Experiment configuration is wrong |
| Policy | Policy rollout retains bound across seeds/speeds | Data/reward is insufficient for the goal |

## Known risks and explicit non-goals

- Changing only `config_go2.timer_t` is insufficient. The generator currently
  hard-codes moving duty factor, and data correctness/quality still needs proof.
- Lowering duty factor creates planned flight windows but may expose solver,
  contact-model, torque, or posture limitations.
- MPX's high pitch/orientation penalties were tuned to resist tilt and may
  conflict with bounding motion.
- The nominal MPX model and rollout model differ. Do not present DR data as
  generated by a dynamics-matched oracle controller.
- Rejection sampling can bias the DR distribution. Acceptance must be reported
  against sampled parameters.
- Direct replay injection is not behavior cloning. The simplified reward has no
  gait identity, so policy-level bound validation is mandatory.
- Real-robot deployment is outside this data-generation implementation. A later
  deployment phase needs independent torque, joint-limit, impact, estimator,
  latency, and emergency-stop review.
- Reworking MPX's primal-dual solver or the nested solver submodule is not
  required unless a reproducible solver defect blocks the validated gait.

## First implementation milestone

The first milestone should be deliberately small:

1. create the isolated root and MPX branches;
2. add/reset-test the MPX bound phase API;
3. define parity metrics/tolerances and test current one-step and short-rollout
   saved-action replay under nominal dynamics;
4. repeat the parity check with a nontrivial deterministic DR patch;
5. preserve the current rollout if it passes, or change generation to advance
   via a validated `env.step(action)` mapping if it fails;
6. make the selected path pass parity with LPF, per-joint gains, clipping, and
   deterministic DR;
7. generate one rendered nominal paired-contact rollout;
8. save measured/planned contacts and run the bound classifier.

Only after this milestone passes should controller tuning and multi-file pilot
generation begin.

## Plan review record

### Critique pass 1: technical consistency

The first review found three material omissions and corrected them in this
document. Its environment-step remediation is now conditional on failing the
empirical parity gate:

1. switching collection to `env.step()` would otherwise have lost the current
   per-simulation-substep non-foot-contact check;
2. the LPF inversion and the stop condition for an unusable MPX-to-action
   mapping were not explicit enough;
3. accepted direct files need pickle-free loading and a fingerprint of the
   effective controller configuration, not provenance based only on Git commits.

The revision adds shared in-environment substep diagnostics, the exact current
LPF inversion, a no-go decision for a failed action mapping, strict loader
behavior, and the expected implementation surface.

### Critique pass 2: executability and leakage control

The second review found that the draft's nominal pilot both derived and applied
its own gait thresholds, which would make acceptance circular. It also left
pilot sizes and phase-to-phase stop conditions too implicit, and a fresh
worktree would not contain this currently untracked plan automatically.

The revision separates calibration, nominal acceptance, and DR seed ranges;
adds G0-G6 evidence gates; proposes bounded commissioning matrices without
presenting them as statistical guarantees; makes the initial training comparison
three seeds per condition; and states how to transfer only this plan into the
clean feature worktree. Bulk storage remains an explicit pre-G6 decision because
no artifact destination can be verified from this repository.

### Plan amendment: empirical parity before remediation

The transition section originally treated the difference between the generator's
direct-torque rollout and the environment action path as a defect requiring an
unconditional rewrite. The existing pipeline is operational, and the available
fake-MPC diagnostic does not establish that its mismatch is material for actual
MPX data or learning. The revised plan predeclares tolerances, measures one-step
and accumulated nominal/DR disagreement using actual MPX output, keeps the
current path if it passes, and changes the action path only if it fails.
