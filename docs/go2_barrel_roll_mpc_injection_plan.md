# Go2 Barrel Roll MPC-Injection Implementation Plan

## Status and purpose

This document is the implementation and experiment plan for a simulation-first
Go2 barrel-roll task in MPC-RL. It is intended to be the durable handoff for
future coding agents and human contributors. It records the decisions made
during the design interview, the repository evidence behind the design, the
required file changes, validation gates, runnable workflows to add, and the
fallback order if a gate fails.

Implementation recorded a G3 commissioning pass on 2026-08-06. During the G4
smoke run later that day, seeds 6 and 8 failed the unchanged classifier. The
project owner approved the established accepted-only generation policy for
this gate: seeds advance sequentially, every failed attempt remains in a
manifest, and only classifier-passing rollouts count toward the requested data
total. G3 is accepted under that revised reproducibility policy, and G4 is
complete with ten validated smoke files. The schema-v2 100k pilot and its best
checkpoint provide the G6 recovery evidence. G7 passed on 2026-08-07 and
promoted the immutable 1,000-file production dataset. No production policy
exists, and G8 production training has not started.

## Implementation evidence (G0-G5 session)

### G0 — complete

- 2026-08-05 baseline: `conda run --no-capture-output -n mpc-rl pytest -q
  tests/test_velocity_tracking_env.py tests/test_quadruped_model_architecture.py
  tests/test_go2_sysid.py` reported `77 passed, 3 failed, 1 skipped`. The
  failures are the pre-existing reward-weight expectation, stale SAC
  `policy_kwargs["net_arch"]` expectation, and missing
  `deploy/sys_id/report.html` fixture.
- Root was `76c3c9070ae864eb4425a006d8b54904b441af1b`; MPX was
  `249c7323ab0cd13bea2ddb8ed5f252eb9ccde85c`; nested solver was
  `273d78dec439ed270af6f9cb8340af801fefd332`; Gym-Quadruped was
  `6eaa8ea6e0ea6dbe0f37965ed7098b974d6cfbcd`; MuJoCo MPC was
  `17be7ffb29ecee55b7d48012bcd6526eab49daf9`. Runtime: Python 3.11.13,
  JAX 0.6.2, and MuJoCo 3.3.6.
- The pre-existing MPX Go2 config and example rename were committed as
  `2f6a7e5` before root integration. The unrelated untracked policy tree was
  not touched.

### G1 — complete

- `conda run --no-capture-output -n mpc-rl pytest -q
  deps/mpx/tests/test_barrel_roll_config.py` reported `2 passed`.
- `conda run --no-capture-output -n mpc-rl python
  deps/mpx/mpx/examples/barrel_roll.py` produced a finite nominal plan:
  91 iterations, final objective norm squared `545477892.2695315`, final
  constraint norm squared `4.456622800618131e-06`, and 14.22 s elapsed.
- No deviations from the frozen one-second schedule or roll direction.
- 2026-08-05 follow-up diagnosis showed that the finite solve above still used
  Aliengo-era robot parameters: 0.33 m standing height, 0.65 duty factor,
  1.35 Hz step frequency, 0.12 m step height, enabled terrain estimation, and
  +/-40 Nm scalar wrapper bounds. The barrel config now matches the existing
  MPX Go2 config for those robot-specific values, including the 0.27 m Go2
  home height and zero-height world-frame foot references. The barrel reference
  now derives its stance and landing heights from that Go2 height; phase timing
  and roll direction are unchanged.
- Focused configuration validation reports `3 passed`. The corrected offline
  solve is finite but no longer feasible enough to retain the former G1 pass:
  after 100 iterations its final constraint norm squared was
  `9.600926568488823`. A diagnostic 300-iteration run reached
  `4.518660053274493`, so raising the iteration limit alone was not a fix. At
  that point G1 and G3 remained blocked pending a Go2-sysID,
  barrel-specific receding-horizon controller path.
- 2026-08-05 continuation added the dedicated
  `MPCControllerWrapper.run_barrel_roll()` path. It rebuilds the MPC state from
  measured `qpos/qvel` and named foot positions, phase-indexes and terminal-pads
  the fixed barrel reference/contact schedule, shifts the prior `X/U/V` warm
  start, returns the two 100 Hz nodes for the next 50 Hz control interval, and
  records objective, constraint, iteration, finite-status, phase, warm-start,
  and timing diagnostics. It does not call the generic gait reference.
- The initial measured-state solve uses 100 SQP updates. A nominal diagnostic
  remained finite and reached constraint norm squared `0.11592822521924973`, a
  material improvement over the corrected no-sysID offline solve but still far
  from the former `1e-5` feasibility threshold. Repeated unit updates retain the
  staged schedule and shift by two MPC nodes. At that point G1 remained blocked
  on feasibility rather than on a missing API.
- `conda run --no-capture-output -n mpc-rl pytest -q
  deps/mpx/tests/test_barrel_roll_config.py tests/test_barrel_roll_env.py`
  reports `8 passed` after these changes.
- The 2026-08-06 recovery closes G1 with forward-execution evidence. The
  reference now has one global clock, continuous Hermite base state, an exact
  signed full-turn quaternion path, phase-dependent joint/foot targets, and the
  XML's ordered per-actuator limits. Inactive contacts are masked before the
  contact solve. The final `1.40` s reference and measured-state plan execute
  physically in the exact MPX Go2 MuJoCo model with sysID, reaching `6.019205`
  rad progress and passing the unchanged classifier without planned-state
  assignment or non-foot contact. The expanded focused suite reports
  `12 passed`.

### G2 — complete

- `conda run --no-capture-output -n mpc-rl pytest -q
  tests/test_barrel_roll_env.py tests/test_velocity_tracking_env.py -k
  'barrel_roll or disabled_dr_generator or dr_replay_matches or
  nominal_replay_path or generation_smoke or saved_transition_parity or
  direct_transition_injection'` reported `9 passed, 54 deselected`.
- The task is Go2-only and nominal, has 45D/4D observations, fixed 50/200 Hz
  timing, seeded `[0.00, 0.10]` named-joint spread sampling, quaternion-safe
  unwrapped roll progress, and the frozen MVP reward/success semantics.
- The environment now requires `use_go2_sysid=True`, supports the explicit
  zero-spread commissioning reset without changing seeded sampling defaults,
  and snapshots roll progress once per control interval so the progress reward
  covers all four physics substeps.

### G3 — complete under revised accepted-only generation policy

- The first direct commissioning invocation failed before it could execute a
  physics step: `/home/roy/miniconda3/envs/mpc-rl/bin/python
  mpc_rl/planner/gen_traj_data_barrel_roll.py --num-trajectories=1
  --start-seed=0 --max-attempts=1 --output-dir=/tmp/go2_barrel_roll_commission2
  --manifest-filename=one.jsonl` exited 1 in
  `MPCControllerWrapper.runOffline()` with
  `jaxlib._jax.XlaRuntimeError: INTERNAL: cuSolver internal error`.
- Before the Go2 parameter correction recorded under G1, the standalone MPX
  smoke remained successful, which isolated the original `cuSolver` failure to
  the generator process. The corrected Go2-height solve now reopens G1 on
  feasibility independently of that allocator issue. No G4-G8 work was
  launched.
- Setting `XLA_PYTHON_CLIENT_MEM_FRACTION=.25` allows the generator process to
  complete the offline call without the `cuSolver internal error`. This is an
  effective allocator workaround, but concurrent JAX processes were not present
  during the confirming run, so launch contention is not established as the
  root cause. The first task-plant execution then failed the classifier on a
  non-foot ground contact with negligible roll progress.
- The MPX MJX-compatible Go2 model and Gym Go2 plant have matching joint and
  actuator ordering, body masses, and torque ranges, but their nominal joint
  damping/friction, solver options, and collision geometry differ. The accepted
  velocity pipeline handles this boundary by applying the same Go2 sysID patch
  to both models and replanning online. Barrel roll will adopt that same policy:
  enable Go2 sysID on both sides and use a barrel-specific receding-horizon
  controller. Exact XML, collision-geometry, and solver-option reconciliation
  is not an implementation gate.
- The generator now constructs both MPX and Gym with Go2 sysID enabled and calls
  the barrel-specific measured-state update at every 50 Hz boundary. It applies
  the returned two-node segment at 200 Hz, preserves the direct-torque/inverse-
  PD convention, accepts `--nominal-spread-zero`, writes structured rejection
  metrics, and writes trajectory files only after classifier success.
- The current source command
  `XLA_PYTHON_CLIENT_MEM_FRACTION=.25
  /home/roy/miniconda3/envs/mpc-rl/bin/python
  mpc_rl/planner/gen_traj_data_barrel_roll.py --num-trajectories=1
  --start-seed=0 --max-attempts=1
  --output-dir=/tmp/go2_barrel_roll_replanning_warm
  --manifest-filename=one.jsonl --nominal-spread-zero --verbose=1` completed 37
  control steps before rear-left hip ground contact. It reached `3.5300803` rad
  roll progress, minimum base height `0.0891304` m, maximum joint tracking error
  `1.9952675` rad, 31.76% action clipping, 1.80% total-torque saturation, and
  4.79% raw MPX-torque saturation. The shifted-plan constraint norm squared
  grew to `1118157.75`. Excluding the initial compile/warm solve, mean replanning
  time was `8.89` ms; the initial 100-update solve took `17.01` s.
- Commissioning-only solver and height diagnostics did not clear the gate. Ten
  SQP updates per replan failed on a base contact at 3.55 rad and increased
  saturation. One hundred updates per replan became non-finite at 0.42 s. A
  0.33 m stance/landing target failed at 3.48 rad, and the former Aliengo-style
  0.33 m stance plus 0.28 m landing combination failed at 0.71 rad. The checked-
  in task-specific stance and landing constants therefore remain at the Go2
  0.27 m height. Further iteration or height sweeps are not justified by these
  results.
- At that point no rendered rollout or seeded ten-rollout set had passed, G3
  remained blocked, and no G4-G8 work had been launched.
- Post-change relevant regression validation (`test_velocity_tracking_env.py`,
  `test_quadruped_model_architecture.py`, `test_go2_sysid.py`,
  `test_barrel_roll_env.py`, and the MPX barrel config test) reports `85 passed,
  3 failed, 1 skipped`. The three failures are exactly the G0 baseline failures;
  no new failure remains. `git diff --check` passes in both root and `deps/mpx`.
- The recovery selected a `1.40` s horizon and phase-limited controller. It
  replans at 50 Hz through flight and landing; after `0.86` s, once the existing
  five-step stable-contact requirement and current raw four-foot contact are
  measured, it performs one final landed-state MPX solve and torque-executes
  the shifted plan with frozen per-leg `Kp=[20,20,40]` and `Kd=[1,1,2]`.
  This transition rule, the initial 100 updates, later single updates, sysID,
  reference, gains, and limits are identical for all seeds.
- A paired same-state/same-torque trace isolated the first relevant Gym transfer
  mismatch to elliptic versus pyramidal friction cones. Only the barrel task's
  cone convention was aligned to the exact MPX model; no XML strength,
  collision geometry, or broad solver reconciliation changed.
- The rendered zero-spread Gym command passed all 70 controller steps with
  `6.143645` rad progress, `-0.141236` rad final roll, `-0.065546` rad pitch,
  `0.273986` m final height, five stable steps, finite solves, and no non-foot
  contact.
- The required command with `--num-trajectories=10 --start-seed=0
  --max-attempts=10` accepted seeds `0` through `9` in exactly ten attempts.
  Every seed passed the unchanged classifier with finite state/control and zero
  non-foot contacts. Per-seed spread, orientation, height, tracking,
  saturation, contact, and solver metrics are recorded in
  `go2_barrel_roll_g3_mvp_recovery_plan.md`; all artifacts remain under `/tmp`.
- Final relevant regression validation reports `89 passed, 3 failed, 1
  skipped`. The four additional passes relative to the prior `85` count are
  expanded barrel focused tests. The three failures remain exactly the G0
  baseline failures. Root and MPX `git diff --check` both pass.
- The first G4 schema-complete smoke command reused seeds `0` through `9`,
  `max_attempts=10`, the same allocator setting, and the frozen controller and
  classifier. It accepted only 8 trajectories. Seed 6 completed 70 transitions
  but ended with a zero stable-contact streak (`incomplete_roll`); seed 8 made
  a non-foot `geom_38` ground contact at 1.025 s and stopped after 52
  transitions. No environment or MPX source differed from the saved boundary.
  This invalidates the reproducibility precondition for G4; no retry or
  replacement seed was used.
- The project owner subsequently approved the same rejection policy used by
  the existing trajectory pipeline: attempt seeds remain sequential, all
  failures remain visible in per-worker and aggregate manifests, only
  classifier-passing rollouts are saved, and generation continues without
  per-seed tuning until the requested accepted count is reached. The 8/10 run
  is retained as evidence rather than replaced; the G4 continuation starts at
  seed 10.

The coordinated feature branches already exist in the current worktree:

- root: `feature/go2-barrel-roll`, created from
  `4b9783af3f43318fb1d634e07c1bd1358a2db2a3`;
- `deps/mpx`: `feature/go2-barrel-roll`, created from
  `249c7323ab0cd13bea2ddb8ed5f252eb9ccde85c`.

The MPX branch contains intentional uncommitted Go2 configuration work plus the
new receding-horizon implementation and tests. Preserve it. The untracked tree under
`deploy/robots/go2/config/policy/velocity/policies/` is unrelated user data and
must not be modified, staged, or deleted.

## Goal and definition of done

Build a minimal research prototype in which SAC-MPC learns a single, fixed-
direction, one-shot barrel roll from MPX demonstrations in MuJoCo simulation.

The work is done only when all of the following are true:

1. MPX uses the Go2 model, pose, feet, and joint ordering throughout the
   barrel-roll path; no Aliengo task values remain active.
2. A seeded generator runs barrel-specific receding-horizon MPX from each
   sampled Go2 start stance and executes it in the same Gym-Quadruped-based
   plant used by the RL task, with Go2 sysID enabled in both models.
3. Only executed rollouts that pass the frozen barrel-roll success classifier
   are saved as versioned, direct RL transitions.
4. The percentage injector strictly validates and injects the new 45D-policy,
   4D-privileged, 12D-action schema into `TaggedDictReplayBuffer`.
5. `mpc_rl/train.py` can train and evaluate
   `--env_name=quadruped-barrel_roll` with SAC-MPC, 25% MPC replay, no domain
   randomization, and the Go2 sysID patch enabled.
6. A 100,000-step pilot shows increasing held-out barrel-roll success before
   any production training starts.
7. Three 500,000-step production seeds complete, and the median policy success
   rate is at least 80% over 100 held-out initialization seeds per policy. All
   three results are reported; weak seeds are not discarded.
8. Tests, commands, dataset provenance, and user-visible workflow documentation
   are current.

Pure SAC, TD3-MPC, bidirectional rolls, domain randomization, and hardware
deployment are not required for this MVP.

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
| Robustness | Use the existing Go2 sysID patch consistently in MPX and the Gym plant. No DR, observation noise, pushes, or encoder bias. |
| MPC execution | Use barrel-specific, phase-aware receding-horizon replanning at each 50 Hz control step. The generic velocity-tracking `run()` reference path is not valid for this maneuver. |
| Model boundary | Follow the compatibility boundary proven by velocity tracking. Do not require XML/collision/solver parity beyond the existing shared ordering and the consistently applied Go2 sysID patch. |
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
MPX Go2 sysID barrel reference with 50 Hz replanning
  -> execute torques in a Gym-Quadruped-based Go2 task plant
  -> save 50 Hz direct Dict-observation transitions
  -> strict barrel-roll dataset validation
  -> PercentMPCInjectCallback
  -> TaggedDictReplayBuffer
  -> SB3 SAC-MPC with asymmetric actor/critic inputs
```

Relevant current behavior:

- `deps/mpx/mpx/examples/barrel_roll.py` is now the exact-model commissioning
  path. It steps the configured Go2 model with direct torque at 200 Hz and never
  assigns optimized `X` after reset.
- `MPCControllerWrapper.run()` remains the velocity-tracking path. Barrel
  commissioning uses the dedicated measured-state `run_barrel_roll()` path,
  which phase-indexes the fixed schedule and supports shifted fixed-plan tail
  execution after the landing replan latch.
- `deps/mpx/mpx/config/config_barrel_roll.py` points at
  `data/go2/go2_mjx.xml`, uses the Go2 home pose/feet, and loads the model's
  ordered actuator limits.
- `deps/mpx/mpx/utils/mpc_utils.py::reference_barell_roll` defines the current
  global `1.40` s reference. The misspelled internal function name remains
  preserved; the public example filename has already been corrected.
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
| 0.40-0.75 s | Flight |
| 0.75-0.85 s | Landing phase |
| 0.85-1.40 s | Final stance |

The signed desired roll stays at zero through 0.20 s, follows the frozen smooth
full-turn profile from 0.20 to 0.80 s, and then stays at the signed `2*pi`
target.
Use one named constant for the chosen sign in the root task code, assert that it
matches the MPX reference during validation, and save it in every dataset file.

At 50 Hz, a current commissioning rollout contains 70 transitions. At 200 Hz it
contains 280 physics steps. The environment classifies success or incomplete
failure on its 70th transition; the Gym `TimeLimit` remains a safety wrapper
rather than the normal source of episode completion.

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
12D normalized residual action, and the existing 5 Hz absolute-target LPF.
Historical schema v1 froze `action_scale=0.5`. The G6 evidence below showed
that value did not represent the commissioned inverse-PD demonstration actions
adequately, so the authorized schema-v2 recovery freezes `action_scale=2.0`.
This is one versioned replacement, not a hyperparameter sweep.

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

The schema-v2 selection rule was declared from demonstration residuals only:
at most 1% aggregate clipping and at most 2% clipping in every accepted file.
The smallest scale satisfying both is `1.900524783`; rounding upward to one
decimal place selects `2.0`. Re-encoding the 84,000 saved schema-v1 residuals
at `2.0` projects 0.5381% aggregate clipping and 1.6667% worst-file clipping.
No held-out policy-evaluation seed informed this choice. Because the virtual
PD target envelope is broader than the Go2 joint ranges, schema v2 must also
pass a non-learning finite-state and torque-limit smoke before dataset
generation. The virtual target is not clamped; the unchanged actuator torque
limits remain the physical control bound.

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
4. Add solve diagnostics without breaking the example's current four-value
   unpacking. The generator needs iteration count, final objective/constraint
   norms if available, finite-status, and elapsed time.
5. Ensure all barrel-roll construction passes `use_go2_sysid=True` explicitly,
   including both the MPX model and the Gym execution plant.
6. Add a barrel-specific receding-horizon entry point. It must retain the fixed
   maneuver reference/contact schedule, index it by current phase, initialize
   from the measured plant state, shift the previous solution for warm start,
   and return the next control segment. Do not route this through the generic
   velocity-reference `run()` implementation.
7. Make the example finite and usable as a smoke command instead of requiring an
   endless viewer loop for validation. Rendering may remain optional.

Validation:

- config/model name and all named IDs resolve;
- reference has `N+1` states, `N` controls, finite values, the intended contact
  stages, one signed turn, and the exact 1.0 s horizon;
- a nominal solve returns finite `X` and `U` with Go2 sysID enabled;
- repeated barrel-specific updates preserve the maneuver phase/contact schedule
  and return finite controls from perturbed measured states;
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
4. Force Go2, flat terrain, disabled DR, disabled pushes/noise, and enabled Go2
   sysID. Reject unsupported robot names rather than silently constructing
   another robot.
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

### Gate G3: execute receding-horizon MPC in the Go2 task plant

Proposed file:

- `mpc_rl/planner/gen_traj_data_barrel_roll.py`

Reuse small helpers from `gen_traj_data_mpx_dr.py` where their semantics match;
do not add barrel-roll modes to the DR generator.

Per attempt:

1. Seed the shared spread sampler and construct `QuadrupedBarrelRollEnv` with
   no DR and `use_go2_sysid=True`.
2. Construct the shared-process MPX wrapper with `use_go2_sysid=True`, copy the
   exact initial Gym `qpos/qvel`, and initialize the fixed barrel-roll reference.
3. At each 50 Hz control boundary, call the barrel-specific replanning entry
   point with the measured Gym state and current maneuver phase. Warm-start
   from the shifted previous solution, retain the fixed phase/contact schedule,
   and apply the returned first control segment for the next four 200 Hz physics
   steps. Record solve timing and iteration count; rendering/commissioning may
   run slower than real time, but the achieved update rate must be measured.
4. Follow the frozen commissioned convention:
   `tau = U + Kp*(q_des - q) - Kd*dq`, clipped to the Go2 plant limits, with
   per-leg `Kp=[20,20,40]` and `Kd=[1,1,2]`. Save the gains as provenance.
5. Compute the inverse-PD residual action from the first actually applied torque
   of each 50 Hz transition. Record unclipped and clipped actions, but inject only
   the clipped action.
6. Mirror environment bookkeeping once per control transition and use the same
   environment reward and success classifier used online.
7. Check non-foot contacts after every physics substep. Allow ordinary foot
   touchdown sequences.
8. Reject non-finite solver output, invalid initialization, non-foot contact,
   incomplete rotation, unstable landing, premature viewer closure, and fewer
   than the configured 70 transitions.
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

Use one MPX wrapper/JIT compilation per process, reinitialize it per seed, and
replan at each control boundary. File names must include task, schema version,
direction, seed, and episode length.
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
classifier. If optimized `X` looks correct but executed rollouts fail, inspect
replanning, phase alignment, tracking gains, saturation, and contact timing on
this gate; do not require broader MPX/Gym XML reconciliation and do not generate
data.

### Gate G4: define and validate schema v1

**Status (2026-08-06): complete under the approved accepted-only policy.** The
smoke set contains ten strictly validated schema-v1 files and 700 direct
transitions. Accepted seeds are `0,1,2,3,4,5,7,9,10,11`; seeds 6 and 8 remain
as rejected records in the aggregate manifest. Seeds 10 and 11 both passed on
their first attempts without tuning, so the complete history is 10 accepted
files from 12 sequential attempts.

The focused schema suite reports `19 passed`; it covers nonzero CLI failure and
explicit schema/task/dimension/timing, non-finite, action, adjacency, phase,
reward, done, contact, rotation, and landing corruptions. The combined relevant
suite reports `108 passed, 3 failed, 1 skipped`; the three failures are exactly
the recorded G0 baseline failures. All ten entries in `checksums.sha256` verify.
The ignored smoke artifacts are under `data/go2_barrel_roll/v1_g4_smoke/`:

- `generation_manifest_worker0.jsonl` and
  `generation_manifest_worker1.jsonl` retain all attempts;
- `generation_manifest_aggregate.jsonl` has SHA-256
  `d400f0387cb5d0f7c83507b757085720e7adfed8d3133a9ae70688863fc4147b`;
- `checksums.sha256` has SHA-256
  `d4550837d7b80cc62f2022016d8d0b7de52dcf45127eb6c572843f0638325b91`;
- `dataset_summary.json` records 10 files, 700 transitions, 12 attempts, two
  rejections, and effective-config SHA-256
  `b94f928e5f9d090ef458b7c8cb2949531040645e07a2318554f6d08967036a94`.

G5 was subsequently authorized and completed as recorded under its gate below.

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

- `policy_obs`, `next_policy_obs`: `(70, 45)`;
- `privileged_obs`, `next_privileged_obs`: `(70, 4)`;
- `actions`: `(70, 12)`;
- `rewards`: `(70,)`;
- `terminated_ctrl`, `truncated_ctrl`: `(70,)`.

Required physics/controller arrays:

- `qpos`: `(19, 281)` and `qvel`: `(18, 281)`;
- `tau_applied`, `tau_mpx`, `q_des`: `(12, 280)`;
- per-update `X`, `U`, or an equally sufficient finite replanning record;
- phase, desired roll, measured roll progress, contacts, clipping, saturation,
  and success metrics at their documented rates.

Required scalar/provenance fields:

- integer `schema_version=1` and task ID `go2_barrel_roll`;
- robot, roll direction, rollout seed, sampled spread, and failure-free success;
- schedule, success thresholds, reward configuration, timing, action scale, LPF
  configuration, PD gains, tracking gains, and torque limits;
- explicit `domain_randomization=disabled` and `go2_sysid_enabled=true`;
- replanning frequency, warm-start policy, and per-update solve diagnostics;
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

**Status (2026-08-06): complete.** `PercentMPCInjectCallback` now sorts its
file set, validates every schema-v1 barrel archive before queuing data, loads
strict task files without pickle, and refuses torque/legacy fallback for
`go2_barrel_roll`. Direct replay preserves distinct transitions across vector
slots, `source=1`, terminal success/failure metadata, and SB3 timeout semantics.
The gate test preloads 210 RL transitions, injects all 70 saved barrel tuples
across two vector slots, verifies bit-exact policy/privileged/done values and
exact actions/rewards after SB3's standard float32 replay conversion, and
reaches exactly 25% MPC composition without entering velocity torque replay.

Training now preserves `velocity_tracking` as the default, routes only
`barrel_roll` to `QuadrupedBarrelRoll-v0`, and rejects unknown tasks or barrel
options that violate Go2/SAC-MPC/direct-25%/no-DR/sysID requirements. The dry
routing command constructed the 45D actor, 4D privileged observation, 12D
action, and 61D critic input without taking a training step. `config.json`
contains the frozen task/schema/direction, schedule, reset, reward, success,
timing, action/LPF/PD, data, DR, sysID, and evaluation contracts.

Barrel evaluation uses the reserved seeds `1000000` through `1000099` and
selects the best model strictly on success-rate improvement; mean reward is
diagnostic only. Videos are seed-labelled and do not set velocity commands.
TensorBoard logging covers phase/progress/error, all three reward components,
contact/stability, terminal outcome and failure reason, clipping/saturation,
and actual MPC percentage.

The required combined suite reports `129 passed, 3 failed, 1 skipped`; the
three failures are exactly the recorded G0/G4 baseline failures. A focused G5
run excluding only the known stale 512-layer assertion reports `46 passed, 1
deselected`. Root and MPX `git diff --check` pass. No G6 data generation or
training was run.

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
2. Force robot Go2, disabled DR, enabled Go2 sysID, the barrel-roll reward, and
   the 70-step horizon for this task. Fail on incompatible requested options
   rather than silently training a different task.
3. Keep SB3 asymmetric SAC and `TaggedDictReplayBuffer`. The actor remains 45D;
   SAC critic first-layer input becomes `45 + 4 + 12 = 61`.
4. Serialize the task ID, schema version, direction, schedule, reset range,
   reward config, success thresholds, timing, action/LPF/PD config, data path,
   25% target, disabled DR, and enabled Go2 sysID into `config.json`.
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

**Status (2026-08-06): BLOCKED; earliest affected gate G2 (action
representability).** Preconditions held at root `e8b0f2ca` and MPX `6bc496d6`.
The required suite reproduced only the three recorded baseline failures
(`129 passed, 3 failed, 1 skipped`) before G6 work. The accepted-only generator
then produced 100 files/7,000 transitions from 134 sequential attempts, seeds
12 through 145. Strict validation reported 100 valid and zero corrupt files.
The 34 rejected attempts remain in the aggregate manifest: 19
`incomplete_roll`, nine `non_finite_solver_output`, four
`non_foot_ground_contact:geom_18`, and two
`non_foot_ground_contact:geom_42`. The aggregate manifest SHA-256 is
`d46c4ec9ba05885ece47bcdc50ef61958af5a02f92e590aaac88000576484f77`;
the checksum-index SHA-256 is
`f3f70eccb300d4a50042130e2592744e29bce8054d84893e3de08335dcf4b1a0`.
Every checksum passes, and every accepted demonstration still passes the
unchanged classifier.

G6 added fail-fast finite-signal diagnostics, durable step-zero/periodic
100-seed evaluation history, and category-level TensorBoard failure metrics
while retaining exact geom-specific reasons in JSONL. An initial pilot attempt
stopped at 19,964 steps because SB3's console formatter truncated two distinct
geom-specific metric names to the same display key. The category metric fix was
tested and the failed run was preserved; no training parameter changed.

The clean rerun
`quadruped-barrel_roll-SAC-MPC-20260806-212611-percentage-25pct-go2-barrel-roll-v1-g6-pilot-rerun1`
completed exactly 100,000 steps with the frozen seed-1 settings. All 100,000
environment transitions, 25,000 Q batches, and 22,500 post-learning training
snapshots were finite. Final replay composition was `25.0014999700006%` MPC.
However, fixed-seed success was 0% at step zero and at every 10k evaluation
through 100k. The final post-training 100-seed evaluation was also 0% success
(85 `geom_23`, 12 `geom_50`, two `geom_47`, and one `geom_26` non-foot
contacts). Policy roll progress ranged from `-1.6492` to only `2.8821` rad.
Critic loss reached `6,174,120.21875`, Q values ranged from `-3,004.37` to
`12,598.17`, and the entropy coefficient ranged from `0.08456` to `4.27058`;
these values remained finite but show unstable value learning.

The fallback audit rules out demonstration execution and injection before
changing learning configuration. All accepted controller executions reach
`5.9998` to `6.2678` rad and land with at least the required five-step stable
contact streak; torque saturation remains low. In contrast, every accepted
file clips more than 20% of saved inverse-PD residual actions. Mean action
clipping is `22.8524%` (maximum `29.4048%`), mean per-file maximum unclipped
action magnitude is `4.5856`, and the maximum is `5.5222` against the saved
`[-1, 1]` range. This is concrete evidence that the initial
`action_scale=0.5` representation assumption failed before SAC hyperparameter
or reward tuning is justified.

The earliest owning gate is G2 because action scale and the online residual
interface are part of the frozen environment/control contract. Per the fallback
order, the next authorized work must deliberately increase one global action
scale, create a new schema/dataset version, and rerun the affected G2-G6
validation chain. Do not mix scales, tune SAC, add reward terms, or begin G7.
After the diagnostics changes, the combined relevant suite reports
`133 passed, 3 failed, 1 skipped`; the failures are exactly the recorded
baseline failures. G6 therefore has the legitimate `BLOCKED` outcome, not a
forced pass.

**Authorized recovery (2026-08-06): schema v2 with
`action_scale=2.0`.** Preserve all schema-v1 data and the blocked pilot above.
Schema v2 keeps the controller, direct-torque transition semantics, 5 Hz LPF,
PD gains, reward, observations, success classifier, reset distribution, SAC
configuration, 25% injection target, training seed, and held-out seeds
unchanged. The affected validation chain is deliberately narrow:

1. G2: prove the v2 environment freezes scale 2.0, rejects other barrel scales,
   and remains finite and torque-limited under deterministic extreme-action
   pulses.
2. G1/G3: rerun focused MPX regression evidence and one nominal plus ten seeded
   controller executions. Do not tune the controller; direct torque execution
   should remain physically unchanged while saved actions and previous-action
   observations change.
3. G4: create ten strictly validated schema-v2 smoke files. Retain v1
   validation support and reject mixed-version aggregation.
4. G5: prove exact v2 direct replay, strict schema routing, configuration
   serialization, and explicit v1 rejection for the v2 training path.
5. G6: generate a fresh 100-file v2 dataset from new non-held-out seeds and
   rerun the same seed-1 100k pilot, including step-zero and every-10k evaluation.

If the non-learning smoke fails, stop and amend the action-interface fallback;
do not try additional scales ad hoc. G7 remains prohibited until the v2 G6
retry passes.

**Schema-v2 recovery validation through G5 (2026-08-06): complete.** The
barrel environment freezes scale 2.0 and rejects scale 0.5; deterministic
single-step extreme-action pulses remain finite and actuator-torque-limited.
The focused environment/data/routing suite reports `60 passed, 1 deselected`,
where the deselection is the documented stale SAC-MPC `net_arch` assertion.
The combined required suite reports `139 passed, 3 failed, 1 skipped`; the
three failures are exactly the recorded reward-weight, stale `net_arch`, and
missing sysID report baseline failures. The focused MPX suite reports `8
passed`.

The nominal seed-146 zero-spread controller execution remained physically
successful with `6.146901` rad progress, 18 stable-contact steps, finite solves,
no non-foot contact, and 0.2381% saved-action clipping. The schema-v2 smoke set
then accepted ten trajectories from 14 sequential attempts, seeds 147 through
160. Accepted seeds are `147,148,149,150,153,154,156,157,158,160`; the four
manifest-visible rejections are two `incomplete_roll`, one
`non_finite_solver_output`, and one `non_foot_ground_contact:geom_18`. All ten
files validate. Mean action clipping is 0.4524% and worst-file clipping is
0.7143%. The aggregate manifest SHA-256 is
`3ea9734604da49d82c45877423d22a4e0fac3495ce200991eda40ab54dc91cec`;
the checksum-index SHA-256 is
`904fd6b05f2b033c451ea068fd8a05ab238464188f5ff45b4d2da24428238136`.

The v2 training dry route constructed the 45D/4D environment and SAC-MPC model
and serialized schema 2, scale 2.0, 5 Hz LPF, and the v2 smoke path. Strict
tests preserve historical v1 validation, reject v1 from the current v2
injection path, reject mixed-version aggregation, and prove exact v2 direct
replay. G6 may now generate a fresh 100-file v2 dataset starting after seed
160; no G7 work is authorized.

The fresh schema-v2 G6 dataset is complete under
`data/go2_barrel_roll/v2_g6_pilot/`: 100 accepted files/7,000 transitions from
148 sequential attempts, seeds 161 through 308. Strict validation reports 100
valid and zero corrupt files. The 48 manifest-visible rejections are 32
`incomplete_roll`, six `non_finite_solver_output`, seven
`non_foot_ground_contact:geom_18`, two `geom_11`, and one `geom_38`. Mean
action clipping is 0.5060% and worst-file clipping is 1.6667%, satisfying the
predeclared aggregate/worst-file limits. The aggregate manifest SHA-256 is
`9c3674429f6d9b94548dcb39e801d258120374a5de0a619b6ef09c4518b96b35`;
the checksum-index SHA-256 is
`e3520b071d34d2479bfa09495faa99264460633c0b170aa6215b012d3aeacd69`.
The v2 G6 training retry is now authorized with no other configuration change.

**Schema-v2 G6 retry status (2026-08-06): PASSED, with checkpoint
instability.** The run
`quadruped-barrel_roll-SAC-MPC-20260806-223427-percentage-25pct-go2-barrel-roll-v2-g6-pilot`
completed exactly 100,000 environment steps with seed 1, four environments,
25% direct injection, and the unchanged SAC/reward/LPF/controller/classifier
settings. All 100,000 environment transitions, 25,000 Q batches, and 22,500
post-learning snapshots were finite. Final replay composition was
`25.0014999700006%` MPC. Critic loss ranged from `0.1185` to `601.1354`, Q
values from `-190.4914` to `407.8657`, entropy coefficient from `0.004001` to
`0.999550`, and observed training roll progress from `-8.5387` to `15.4878`
rad. This is materially more stable and more rotational than the schema-v1
pilot, though over-rotation remains visible.

Fixed held-out success was 0% at steps 0 through 70k, rose to 15% at 80k,
then regressed to 0% at 90k and 100k. The success-selected 80k checkpoint was
preserved and an independent reload/evaluation reproduced exactly 15/100
successes, with 83 `incomplete_roll` and two
`non_foot_ground_contact:geom_30` failures. The periodic 100k evaluation was
0% with 87 incomplete rolls and 13 `geom_53` contacts; the separate final-model
evaluation was also 0%, with 99 incomplete rolls and one `geom_17` contact.
G6 therefore passes its declared requirement of clear held-out improvement
above the initial level before the production-scale experiment; it does not
claim stable convergence or the deferred 80% production threshold. Preserve
the full evaluation history, diagnostics summary, final model, and best model
under the run directory. G7 is now dependency-unblocked but was not started in
this recovery session.

Generate 100 accepted trajectories only after G0-G5 pass. This gives 7,000
distinct direct transitions.

Run one SAC-MPC seed for 100,000 environment steps with production settings
except duration and dataset size:

- 25% percentage injection;
- existing quadruped SAC-MPC network and optimizer defaults initially;
- 50/200 Hz task timing;
- schema-v1 historical run: `action_scale=0.5`; schema-v2 recovery run:
  `action_scale=2.0`; both retain the existing 5 Hz online LPF;
- no DR and Go2 sysID enabled;
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

**Status (2026-08-07): PASSED.** The user authorized root `6fdd5bee` as the
frozen generation revision after the initial revision audit found that it was
one unrelated visualizer commit beyond `9e1c13d5`. Frozen submodule revisions
were MPX `6bc496d6`, primal-dual iLQR `273d78de`, gym-quadruped `6eaa8ea6`,
and MuJoCo MPC `17be7ffb`. The required root suite reproduced exactly the three
permitted baseline failures (`131 passed, 3 failed, 1 skipped`), and the
focused MPX suite reported `8 passed, 7 deselected`. Strict revalidation of all
100 G6 files and their checksum index reproduced 100 valid files, 7,000
transitions, 148 attempts, 48 rejects, clipping statistics, diagnostics,
evaluation history, and the two recorded hashes before production generation.

The four generator invocations used the arguments shown below, with starts
`10000`, `110000`, `210000`, and `310000`, 250 accepted files per worker,
2,500-attempt limits, and unique worker manifests. Each invocation was prefixed
with `/usr/bin/time -v env` to capture wall time and host memory. Worker 0 ran
alone first.
Its initial process used 8,548 MiB of GPU memory; total device use including
the desktop was 11,612 MiB of 32,607 MiB. Two workers used 20,231 MiB total
with 11,857 MiB free. Four workers would have exceeded device capacity, so
concurrency was capped at two. Every process was a separate generator process
and therefore owned its own MPX wrapper; no wrapper was shared.

The G4/G6 resource audit found one-time JAX compilation maxima of 16.31 and
15.89 seconds, single-worker replans near 9 ms, 67.6% G6 acceptance, and mean
accepted-file size near 2.7 MB. Production workers completed as follows:

| Worker | Attempts | Accepted | Rejected | Acceptance | Wall time | Mean replan | JAX/solve max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 372 | 250 | 122 | 67.20% | 16:02 manifest window | 14.67 ms | 16.31 s |
| 1 | 359 | 250 | 109 | 69.64% | 15:58 | 15.29 ms | 17.60 s |
| 2 | 343 | 250 | 93 | 72.89% | 16:02 | 15.77 ms | 17.60 s |
| 3 | 361 | 250 | 111 | 69.25% | 16:13 | 15.32 ms | 17.57 s |

Observed peak host RSS was 3,054,396 KiB (2.91 GiB). Each accepted NPZ averages
2,686,718 bytes; the 1,000 files total 2,686,718,482 bytes. Two-worker GPU
contention increased replan time relative to the single-worker G6 evidence but
did not cause resource failure, solver sharing, or a changed configuration.

Aggregation used:

```bash
conda run --no-capture-output -n mpc-rl python \
  mpc_rl/planner/barrel_roll_dataset.py \
  data/go2_barrel_roll/v2_staging --aggregate

(
  cd data/go2_barrel_roll/v2_staging &&
  sha256sum --check --strict checksums.sha256
)
```

Strict aggregation and a separate NPZ/JSONL/checksum reconciliation proved
exactly 1,000 unique accepted schema-v2 files/seeds and 70,000 direct
transitions from 1,435 unique attempts. Each worker contributed exactly 250
accepted files. All files share effective configuration SHA-256
`8013d45f6ea6540242a0aa49f5cf728d736e08e451ae498edb697703da803c0d`,
root revision `6fdd5bee`, and generator-source SHA-256
`7b52aa0b718b3c3b53ee89408e11ac95f9f6eab09b17e417f505e3a42a3c9067`.
No accepted or attempted production seed overlaps seeds 0 through 308 or the
held-out range `1000000` through `1000099`; no separate resource-smoke seeds
were used.

All 435 rejected attempts remain in the worker and byte-for-byte reconciled
aggregate manifests: 283 `incomplete_roll`, 67 `non_finite_solver_output`, 39
`non_foot_ground_contact:geom_18`, 11 `geom_38`, six each `geom_11` and
`geom_42`, five `geom_14`, four `geom_23`, three `geom_8`, two each `geom_30`,
`geom_35`, and `geom_50`, and one each `geom_26`, `geom_29`, `geom_47`,
`geom_53`, and `geom_54`. Mean action clipping is
`0.004986904761904763` (0.4987%) and the worst accepted file is
`0.017857142857142856` (1.7857%), below the frozen 1% aggregate and 2%
per-file limits. The aggregate-manifest SHA-256 is
`8d67b8ab296bc4bd56cffac69e34a44f2b8da922b203e0b0b9c1b4ad931b5491`;
the checksum-index SHA-256 is
`153334c544fec09e8aee4fa74223bde0f5b1c6b4f0018ce1e65328fbf0ccee70`.
Every indexed NPZ checksum verifies before and after promotion.

During generation, the separately authorized lateral-push evaluator work made
the root dirty for 559 later files; 441 earlier files record a clean root.
Those edits did not touch the generator or its dependencies: every file has
the same root commit, generator-source hash, schema, and effective configuration
hash. The unrelated changes were validated and committed separately as
`6d370fcf`, outside the G7 evidence commit.

After every validation above passed and the destination absence check
succeeded, promotion used the required same-filesystem rename:

```bash
set -euo pipefail
test ! -e data/go2_barrel_roll/v2
mv -- data/go2_barrel_roll/v2_staging data/go2_barrel_roll/v2
```

The promoted ignored directory is the immutable production dataset. G8 was not
started; it requires a new instruction.

After the 100k pilot passes, collect 1,000 accepted trajectories (70,000 direct
transitions). Run multiple terminals with disjoint, generously separated seed
ranges and unique manifest names. The implemented CLI permits four workers to
target 250 successes each, subject to the resource audit below:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=.25 conda run --no-capture-output -n mpc-rl \
  python mpc_rl/planner/gen_traj_data_barrel_roll.py \
  --num-trajectories=250 --start-seed=10000 --max-attempts=2500 \
  --output-dir=data/go2_barrel_roll/v2_staging \
  --manifest-filename=generation_manifest_worker0.jsonl

XLA_PYTHON_CLIENT_MEM_FRACTION=.25 conda run --no-capture-output -n mpc-rl \
  python mpc_rl/planner/gen_traj_data_barrel_roll.py \
  --num-trajectories=250 --start-seed=110000 --max-attempts=2500 \
  --output-dir=data/go2_barrel_roll/v2_staging \
  --manifest-filename=generation_manifest_worker1.jsonl
```

Workers 2 and 3 use starts `210000` and `310000` with manifest suffixes
`worker2` and `worker3`. The CLI is implemented. Start with a measured
single-process attempt and increase concurrency only if GPU memory and solver
behavior remain safe; four concurrent workers are an example, not an
acceptance requirement.

Before adding workers, measure JAX compilation time, per-update and per-attempt
solve time, GPU memory, success rate, and file size from the 10- and 100-file
stages. Each process owns one MPX wrapper; do not share a compiled wrapper
between processes.

After collection:

1. verify there are exactly 1,000 unique accepted seeds/files;
2. run strict validation on every file;
3. aggregate every attempt manifest and report rejection reasons;
4. create checksums and the dataset summary;
5. promote by atomic directory rename from staging to
   `data/go2_barrel_roll/v2` after proving that target does not already exist;
6. never add `.npz` files to Git.

Gate PASS requires exactly 1,000 unique accepted schema-v2 seeds/files and
70,000 direct transitions; strict validation of every file; no generation seed
overlap with calibration, smoke, pilot, or held-out evaluation seeds; complete
worker and aggregate attempt manifests with rejection reasons; a verified
checksum index and dataset summary; the predeclared clipping limits of at most
1% aggregate and at most 2% in every accepted file; and atomic promotion to
`data/go2_barrel_roll/v2`. The promoted directory is immutable for the
production experiment. Any action scale, reward, schema, controller, or
success-threshold change creates a new dataset version and returns work to the
earliest affected gate.

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
  --data_dir=data/go2_barrel_roll/v2 \
  --domain_rand=False \
  --domain_rand_config_type=disabled \
  --use_go2_sysid=True \
  --seed=<seed> \
  --suffix=go2-barrel-roll-v2
```

The flags in this command are implemented and G7 has frozen the promoted
checksum index, but G8 was not started and still requires a new instruction.

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
| Schema | Strict current-v2 validation, historical-v1 support, mixed-version rejection | Fix generator/validator |
| Injection | Exact direct tuples, MPC tags, multi-env batching, 25% | Fix loader/callback |
| Model | SAC actor 45D, critic input 61, finite update | Fix routing/policy config |
| Pilot | Increasing held-out success by 100k | Use fallback ladder |
| Production data | 1,000 unique valid v2 files, 70,000 transitions, clipping limits, manifests, and checksums | Regenerate failed shards |
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
2. **Plan valid but task-plant execution fails:** verify sysID is enabled on
   both sides, then inspect replanning/phase alignment, node-rate mapping,
   tracking error, feedback gains, torque saturation, and contact timing. Tune
   only on commissioning seeds; do not make exhaustive XML reconciliation a
   prerequisite.
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
   reward term. TD3-MPC, torque actions, LPF removal, and DR remain outside the
   initial fallback ladder.

Every fallback that changes observations, action scale, reward semantics,
controller behavior, or success thresholds invalidates the existing dataset
version and must be reflected in config serialization and documentation.

## Documentation and commit sequence

Update after the corresponding behavior exists:

- `README.md`: task name, generator/validator commands, data location, training
  and evaluation commands, 50/200 Hz timing, no-DR/Go2-sysID scope, and artifact
  outputs;
- this plan: mark gates complete with exact validation results and deviations;
- `docs/HZ_CONTROL_REFERENCE.md`: add the barrel-roll row only if the new task
  changes or clarifies user-visible timing semantics;
- `docs/low_pass_filter_implementation.md`: state explicitly that online
  barrel-roll actions use the LPF while MPC demonstration generation follows
  the existing direct-torque/inverse-PD pipeline;
- MPX README/example comments: Go2 barrel-roll invocation and finite smoke use;
- the new experiment script: exact dataset, schema, percentage, disabled DR,
  enabled Go2 sysID, seeds, and budgets.

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
- random terrain, DR, pushes, observation noise, or encoder bias;
- exhaustive MPX/Gym XML, collision-geometry, or solver-option reconciliation;
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
