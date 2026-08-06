# Go2 Barrel-Roll G3 MVP Recovery Plan

**Status (2026-08-06): complete.** The rendered zero-spread rollout and the
single-attempt seeds `0` through `9` gate pass the unchanged classifier. The
commissioning evidence and frozen controller are recorded below.

## Purpose and endpoint

This is a narrow recovery plan for the blocked G3 controller gate in
[`go2_barrel_roll_mpc_injection_plan.md`](go2_barrel_roll_mpc_injection_plan.md).
That document remains the source for implementation history, commands, locked
project decisions, and recorded failure evidence. This plan covers only the
work required to produce a physically executed Go2 barrel roll.
Its permissions to change phase timing, total horizon, joint/foot references,
and replanning mode are explicit recovery exceptions agreed in the design
interview; they do not reopen unrelated decisions in the original plan.

The recovery is complete only when both of these pass the existing, unchanged
success classifier in `mpc_rl/envs/barrel_roll_env.py`:

1. one zero-spread rollout is visibly rendered to completion in the current
   `QuadrupedBarrelRollEnv`; and
2. ten consecutive rollouts using seeds `0` through `9` and the existing
   `[0.00, 0.10]` hip-spread sampler all succeed, with no retries or replacement
   seeds.

Do not start schema, dataset, injection, training, reward, or policy work. Any
commissioning output remains disposable under `/tmp`.

## Fixed success and execution constraints

- Success still requires the configured full-turn progress, upright roll and
  pitch, minimum base height, five consecutive four-foot stable control steps,
  finite state/control, and no non-foot ground contact.
- Reset remains the Go2 home standing pose plus the existing symmetric hip
  spread. A crouch or preload must be executed after reset.
- The final gate runs in the current Gym-Quadruped-based
  `QuadrupedBarrelRollEnv`, with Go2 sysID enabled.
- MuJoCo physics remains 200 Hz and the controller boundary remains 50 Hz.
- The maneuver must come from an MPX-optimized trajectory and be physically
  executed through direct torque. Assigning planned `X` state into a simulator,
  hand-writing a torque sequence, or substituting another optimizer does not
  pass.
- The execution controller may use a fixed MPX plan with feedback,
  phase-limited replanning, or 50 Hz replanning. Use the smallest mode that
  passes; per-step replanning is not an MVP requirement.
- Do not increase the Go2 actuator limits. MPX optimization and both execution
  plants must use the model's per-actuator ranges: the current XML limits are
  lower for hip/thigh actuators than for knees.
- The maneuver horizon is one global configurable value. Start at 1.0 s and
  increase it only when timeline evidence identifies inadequate preparation,
  rotation, or stabilization time. Freeze one horizon before the ten-seed
  gate; never select it per seed. Keep 100 Hz MPX nodes and 50/200 Hz execution
  timing when the horizon changes.

## Current blocker

The best recorded Gym rollout fails after 37 control steps on rear-left hip
ground contact at 3.53 rad of measured roll, while the reference demands about
5.65 rad. Joint tracking error, residual-action clipping, and shifted-plan
constraint residuals are high. Iteration-count and standing-height trials did
not fix it.

The repository also exposes defects that must be resolved before more tuning:

- `reference_barell_roll()` restarts local phase coordinates, producing state
  discontinuities at stage boundaries, including a lateral-position reset at
  the start of flight;
- its quaternion integration stops one node short of a full turn and then
  replaces the next sample with the identity quaternion;
- it uses the home joint pose throughout push-off, flight, and landing, opposing
  useful crouch, tuck, and landing configurations; and
- `quadruped_wb_obj()` uses a hardcoded symmetric 44 Nm torque penalty boundary
  instead of the Go2 model's per-actuator limits, while physical execution clips
  to the XML ranges.

The offline viewer in `deps/mpx/mpx/examples/barrel_roll.py` is not execution
evidence because it assigns planned `X` directly into MuJoCo.

## Recovery sequence

Execute the following steps in order. Do not tune gains, solver iterations, or
model parameters until Step 1 passes.

### Step 1: repair and validate the maneuver definition

Primary files:

- `deps/mpx/mpx/utils/mpc_utils.py`
- `deps/mpx/mpx/config/config_barrel_roll.py`
- `deps/mpx/mpx/utils/objectives.py`
- `deps/mpx/tests/test_barrel_roll_config.py`
- `mpc_rl/envs/barrel_roll_common.py` only to keep the root task timing aligned

Make the smallest changes needed to produce one coherent reference:

1. Generate all stages from one global node-time vector. Include each boundary
   once and return exactly `N + 1` state/reference nodes for `N` controls.
2. Make base position, linear velocity, quaternion, angular velocity, joint
   pose, and foot targets agree at adjacent stage endpoints. Contact flags may
   switch only at the named stage boundaries; the kinematic state must not reset
   when they switch.
3. Integrate exactly one signed `2*pi` rotation. At roll completion the
   quaternion must be equivalent to upright by sign, without a final snap, and
   angular velocity must transition consistently into stabilization.
4. Replace the single tiled home joint target with the minimum phase-dependent
   crouch/push-off, tuck, and landing targets needed by the optimizer. These are
   references, not simulator assignments.
5. Preserve a standing reset and the fixed roll direction. Phase durations and
   the total horizon may change together as a single checked-in configuration.
6. Replace the objective's hardcoded 44 Nm values on this path with the MPX
   model's ordered per-actuator `actuator_ctrlrange`. Do not edit either Go2 XML
   to raise its limits.

Gate:

- reference, parameter, `X`, and `U` are finite;
- dimensions and timing are internally consistent for the configured horizon;
- boundary tests detect no coordinate restart or quaternion snap;
- accumulated reference rotation is one signed full turn;
- contact switches occur only at configured boundaries and touchdown foot
  targets agree with the landing surface;
- reported torque demand and saturation use the actual per-joint limits; and
- an MPX solve from the zero-spread standing state has finite diagnostics. The
  forward-execution result in Step 2, not planned-state rendering, decides
  physical feasibility.

### Step 2: prove exact-model forward execution

Use the existing MPX Go2 XML from `config.model_path` as a MuJoCo execution
plant, with the same Go2 sysID patch, 0.005 s physics step, initial state,
per-joint actuator limits, and direct controller convention used by G3. Add this
as a commissioning path in the existing barrel generator or MPX example; do
not add a generalized rollout framework.

For the first check, solve one nominal MPX trajectory, then step MuJoCo forward
from the standing state using:

```text
tau = clip(U + Kp * (q_des - q) - Kd * dq, actuator_ctrlrange)
```

Hold each 100 Hz MPX node for two 200 Hz physics steps. After reset, never write
planned `X` into `qpos` or `qvel`.

Record the existing G3 metrics at every controller boundary: measured and
desired roll, base height, joint tracking error, applied/raw MPX saturation,
non-foot contact, contact sequence, and classifier state. Also record the first
time and state at which measured execution materially departs from planned
`X`.

Interpretation and gate:

- If this fixed-plan exact-model rollout fails, the blocker is the reference,
  optimizer dynamics/contact assumptions, or torque-level tracking. Return to
  Step 1 and change only the item implicated by the first divergence.
- Do not proceed merely because planned `X` looks correct.
- Step 2 passes only when the physically stepped exact-model nominal rollout
  passes the unchanged classifier without actuator-limit violations.

### Step 3: select the minimum controller mode

Start with the fixed MPX plan plus feedback that passed Step 2. Keep it if it
also transfers successfully in Step 4. Introduce replanning only if measured
disturbance rejection or the spread variation requires it.

If replanning is needed, reproduce the same exact-model rollout and compare a
cold solve and shifted warm solve from the same captured measured state at the
first failing phase. For each solve, check:

- phase index and first contact row;
- measured `x0` versus the optimizer's first state constraint;
- reference and parameter shift amounts;
- `X/U/V` warm-start shift amounts;
- objective and constraint norms and finite status; and
- the first returned `q_des` and torque segment.

Fix phase indexing, horizon padding, measured foot-state construction, or warm
start only when that comparison identifies it. Do not sweep solver iterations:
the existing 1/10/100-update evidence already shows that iterations alone do
not solve the failure.

Gate: the chosen controller mode passes Step 2 and has one deterministic phase
and warm-start convention. Remove or leave unused diagnostic alternatives; do
not build a runtime controller-selection layer.

### Step 4: transfer the passing controller to the Gym plant

Run the exact Step 2 controller and initial state in
`mpc_rl/planner/gen_traj_data_barrel_roll.py` against
`QuadrupedBarrelRollEnv`. Compare the exact-model and Gym traces from time zero
using base pose/velocity, joint position/velocity, applied torque, contact
sequence, roll progress, and first non-foot contact.

Change only the first evidenced transfer mismatch. Permitted changes are the
barrel reference, chosen execution mode, feedback gains, and a specific
MPX-to-Gym control/model interface value proven responsible by the paired
trace. The final plant remains `QuadrupedBarrelRollEnv`, sysID stays enabled on
both sides, and actuator limits stay unchanged. Do not perform broad XML,
collision-geometry, or solver-option reconciliation.

Use these failure rules:

- Tracking error without saturation permits a focused `Kp/Kd` correction.
- Tracking error with saturation requires a feasible reference/trajectory;
  higher gains cannot create missing torque.
- Insufficient preparation, rotation, or landing-recovery time permits one
  global horizon/phase revision followed by repeating Steps 1 and 2.
- A contact-timing mismatch permits a reference/contact correction only after
  confirming the measured kinematics at that boundary.
- Non-finite or exploding replans return to Step 3; they do not justify model or
  gain tuning.

Gate: one zero-spread Gym rollout passes the classifier and is rendered through
the actual forward-execution loop. Save no promoted trajectory or dataset.

### Step 5: freeze and pass seeds 0-9

Freeze the reference, horizon, controller mode, gains, solver settings, sysID
configuration, and classifier after the nominal rendered pass. Run exactly ten
attempts with default spread sampling and seeds `0` through `9`. Use no retries,
replacement seeds, per-seed parameters, or post-reset state changes.

Gate: all ten attempts pass the unchanged classifier. Report each seed's
sampled spread, roll progress, final roll/pitch error, minimum base height,
stable-contact streak, non-foot contact count, maximum joint tracking error,
and MPX/applied torque saturation. Any failure leaves G3 blocked and must be
fixed by returning to the earliest applicable step, followed by rerunning the
full ten-seed set.

## Focused validation

Use the existing Conda environment and allocator workaround recorded in the
original plan. The relevant checks are:

```bash
conda run --no-capture-output -n mpc-rl pytest -q \
  deps/mpx/tests/test_barrel_roll_config.py \
  tests/test_barrel_roll_env.py

git diff --check
git -C deps/mpx diff --check
```

After the nominal controller passes, rerun the combined relevant regression
suite recorded under G3 in the original plan and compare it with the documented
`85 passed, 3 failed, 1 skipped` baseline. The expanded focused tests add four
passing cases to that count. Do not weaken or skip existing assertions to
obtain a pass.

The commissioning commands should continue to use
`mpc_rl/planner/gen_traj_data_barrel_roll.py`, `--nominal-spread-zero` for the
rendered check, and a temporary output directory. For the ten-seed gate, use
`--num-trajectories=10 --start-seed=0 --max-attempts=10`; this command can reach
ten accepted results only if every seed passes.

## Execution record (2026-08-06)

### Repaired maneuver and exact-model execution

- The reference now uses one global `0.01` s node clock and returns `N + 1`
  coherent state/reference nodes. Hermite position/velocity segments eliminate
  stage-coordinate restarts. A continuous smoothstep quaternion reference
  accumulates exactly signed `2*pi` between `0.20` and `0.80` s.
- The frozen `1.40` s schedule is `0.20` s initial stance, `0.20` s lateral
  support, `0.35` s flight, `0.10` s landing, and `0.55` s final stance. The
  intermediate `1.20` and `1.30` s gates showed completed upright rotations but
  late contact-streak resets; only final-stance time was extended.
- Frozen base values are `0.27` m stance/landing, `0.18` m crouch, `0.38` m
  takeoff, `0.34` m touchdown, and `-0.50` m/s lateral speed. Joint references
  now stage the crouch, asymmetric push-off, tuck, touchdown, and home recovery
  instead of holding the home pose throughout.
- MPX cost, Hessian, dynamics, and execution use the XML actuator order and
  limits (`+/-23.7` Nm hip/thigh and `+/-45.43` Nm calf). Inactive contact rows
  are removed before the contact solve, and optimizer dynamics clip controls to
  those limits. No XML or classifier threshold changed.
- `deps/mpx/mpx/examples/barrel_roll.py` now steps the exact configured MPX
  MuJoCo robot at `0.005` s with Go2 sysID and direct torque. It never assigns
  planned `X` after reset. The final exact-model run passed with `6.019205` rad
  progress, `-0.264365` rad final roll, `0.041064` rad pitch, `0.268196` m final
  height, `0.199754` m minimum height, 13 stable steps, no non-foot contact,
  `0.575766` rad maximum joint error, `3.9286%` raw-MPX saturation, and `3.8095%`
  applied saturation. Final objective/constraint norms squared were
  `250743456.0` and `10216.7939453125`; all solver outputs were finite.

Exact-model command:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=.25 \
conda run --no-capture-output -n mpc-rl -- \
python deps/mpx/mpx/examples/barrel_roll.py \
  --trace-path=/tmp/go2_barrel_roll_exact_trace_stable_latched.npz \
  --verbose=1
```

### Controller selection and Gym transfer

- Fixed-plan feedback did not generate the full rotation. Full 50 Hz replanning
  was required through flight and touchdown. The selected minimum mode is
  phase-limited replanning: beginning no earlier than `0.86` s, once the
  unchanged five-step stable-contact condition and current raw four-foot
  contact are both measured, one final MPX solve is taken from that landed
  state; its shifted plan is then torque-executed with feedback. This is one
  frozen state-transition rule for every seed, not a per-seed parameter or
  retry policy.
- The frozen feedback gains in actuator order are per-leg
  `Kp=[20, 20, 40]` and `Kd=[1, 1, 2]`. The first measured-state solve uses 100
  updates and later replans use one update. Physics remains 200 Hz, controller
  boundaries remain 50 Hz, and MPX nodes remain 100 Hz.
- A paired same-initial-state/same-torque comparison identified the first
  transfer mismatch as contact-cone convention. At `0.20` s, setting only the
  Gym barrel plant from elliptic to the MPX model's pyramidal cone reduced the
  maximum paired `qpos` difference from `0.312439` to `0.268226` and `qvel`
  difference from `5.707658` to `3.706405`. No broad XML or solver-option
  reconciliation was performed.
- The final visible zero-spread Gym rollout passed with `6.143645` rad progress,
  `-0.141236` rad final roll, `-0.065546` rad pitch, `0.273986` m final height,
  `0.201864` m minimum height, five stable steps, no non-foot contact,
  `0.389277` rad maximum joint error, `1.7262%` raw-MPX saturation, and
  `1.6369%` applied saturation. All 70 controller steps and 280 physics steps
  executed; solver finite status and classifier result were true.

Rendered nominal command:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=.25 \
conda run --no-capture-output -n mpc-rl -- \
python -m mpc_rl.planner.gen_traj_data_barrel_roll \
  --num-trajectories=1 --start-seed=0 --max-attempts=1 \
  --nominal-spread-zero \
  --output-dir=/tmp/go2_barrel_roll_gym_rendered_stable_latched \
  --render --verbose=1
```

### Frozen seeds 0-9 gate

The final command made exactly ten attempts and accepted exactly ten results.
No seed was retried or replaced. `Stop` is the measured time of the frozen
landing transition; it varies with physical touchdown while the transition
rule and all controller parameters remain identical. `Min height` is the
minimum over the entire airborne trajectory; all terminal heights were between
`0.271034` and `0.276751` m and therefore cleared the unchanged terminal height
criterion.

| Seed | Spread | Progress | Roll | Pitch | Min height | Stable | Max q err | MPX sat | Applied sat | Non-foot | Stop |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.065046 | 6.161935 | -0.120990 | -0.035449 | 0.202142 | 16 | 0.475543 | 0.014881 | 0.013393 | 0 | 1.02 |
| 1 | 0.096993 | 6.135010 | -0.149085 | -0.033320 | 0.198819 | 17 | 0.666440 | 0.026786 | 0.026786 | 0 | 1.16 |
| 2 | 0.010454 | 6.075248 | -0.207645 | 0.010768 | 0.198158 | 12 | 0.475475 | 0.016071 | 0.013393 | 0 | 1.08 |
| 3 | 0.070697 | 6.097498 | -0.186896 | 0.039139 | 0.216349 | 32 | 0.465232 | 0.020833 | 0.019940 | 0 | 0.86 |
| 4 | 0.022377 | 6.172187 | -0.110728 | -0.022746 | 0.199843 | 22 | 0.486286 | 0.013095 | 0.010417 | 0 | 1.06 |
| 5 | 0.001852 | 6.175561 | -0.106750 | -0.028750 | 0.196848 | 21 | 0.513548 | 0.014881 | 0.013095 | 0 | 1.08 |
| 6 | 0.043966 | 6.147384 | -0.134840 | -0.064413 | 0.210484 | 5 | 0.471374 | 0.014286 | 0.010714 | 0 | 1.02 |
| 7 | 0.051412 | 6.041220 | -0.248439 | 0.068736 | 0.217766 | 6 | 0.636943 | 0.027976 | 0.025595 | 0 | 1.38 |
| 8 | 0.045227 | 6.050019 | -0.232470 | -0.033429 | 0.208221 | 9 | 0.503128 | 0.022619 | 0.019345 | 0 | 1.00 |
| 9 | 0.008973 | 6.115028 | -0.164141 | -0.078032 | 0.203562 | 9 | 0.452208 | 0.021429 | 0.021726 | 0 | 1.16 |

Ten-seed command:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=.25 \
conda run --no-capture-output -n mpc-rl -- \
python -m mpc_rl.planner.gen_traj_data_barrel_roll \
  --num-trajectories=10 --start-seed=0 --max-attempts=10 \
  --output-dir=/tmp/go2_barrel_roll_g3_seeds_0_9_stable_latched \
  --verbose=1
```

Every attempt wrote a disposable trace containing measured/desired roll, base
pose, joint errors, raw/applied torque and per-actuator saturation, contacts and
first non-foot contact, phase/horizon position, objective/constraint norms,
finite status, warm-start shift, replanning state, stable streak, and classifier
result. All ten traces load with `allow_pickle=False` and have the expected 281
state samples and 280 torque samples.

### Validation result

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=.25 \
conda run --no-capture-output -n mpc-rl pytest -q \
  deps/mpx/tests/test_barrel_roll_config.py \
  tests/test_barrel_roll_env.py
```

Result: `12 passed, 1 warning`.

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=.25 \
conda run --no-capture-output -n mpc-rl pytest -q \
  tests/test_velocity_tracking_env.py \
  tests/test_quadruped_model_architecture.py \
  tests/test_go2_sysid.py \
  tests/test_barrel_roll_env.py \
  deps/mpx/tests/test_barrel_roll_config.py
```

Result: `89 passed, 3 failed, 1 skipped, 4 warnings`. The extra four passes
relative to the recorded `85`-pass comparison are the expanded focused tests.
The failures are unchanged: the reward-weight expectation, stale SAC
`policy_kwargs["net_arch"]` expectation, and missing
`deploy/sys_id/report.html` fixture.

Both `git diff --check` and `git -C deps/mpx diff --check` pass. The only
remaining checks are those three documented baseline failures. No G4 schema,
dataset promotion, injection, training, policy, or forbidden policy-tree work
was performed.

## Explicit exclusions

- schema or dataset implementation and promotion;
- promoted trajectory generation; commissioning may write only disposable
  diagnostic output under `/tmp`;
- replay injection, SAC/TD3 training, reward formula/weight changes, or
  observation content/dimension changes;
- relaxed success thresholds or a longer stability allowance;
- increased actuator limits;
- alternate robots, roll directions, randomization, noise, or pushes;
- hand-authored torque playback, planned-state playback, or another optimizer;
- broad XML parity work, architecture refactors, generic diagnostics
  infrastructure, or parameter sweeps; and
- unrelated velocity-task, policy-export, or baseline-test cleanup.
