# Go2 Barrel-Roll 25% MPC-Injection Schema-v4 Recovery Plan

## Status and objective

Status: Gates V4.1 through V4.4 passed. The user explicitly approved all 10
schema-v4 MPC demonstrations on 2026-08-08. The production dataset is
atomically promoted, both fixed 500,000-step training runs completed, and the
automated Gate V4.5 selected-policy review package is ready for visual review.
The user requested 10 success videos for each seed's selected policy even if
neither validation score reaches 80/100; all 20 requested success videos are
present. The sealed final-test seeds remain unopened pending explicit Gate
V4.5 video approval.

The objective is to train one neural policy that performs one Go2 barrel roll
and then stands through the end of the same fixed 2.50-second episode. The
policy must use the existing SAC-MPC percentage-injection pipeline with 25%
MPC replay. There is no handoff to a separate standing controller.

Schema v4 replaces the behaviorally invalid schema-v3 task contract. The
schema-v2 and schema-v3 datasets, campaigns, reports, evaluations, and videos
remain immutable historical evidence. A schema-v3 classifier pass is not
evidence that a rollout satisfies this plan.

Do not commit, push, or open a pull request unless the user explicitly asks.
Never inspect, enumerate, modify, stage, or include
`deploy/robots/go2/config/policy/velocity/policies/`.

### Execution record: MPC demonstration approval pause

- The user reported that the demonstrations worked perfectly and explicitly
  approved all 10 videos on 2026-08-08. This satisfies the visual portion of
  Gate V4.2 and authorizes Gate V4.3; it does not authorize opening the sealed
  final-test seeds before the separate Gate V4.5 approval.

- The versioned schema-v4 task, environment termination and final-hold
  classifier, reward and logging fields, strict dataset validator, generator,
  accepted-state renderer, demonstration report, and direct-injection schema
  routing are implemented. Historical schema-v1 through schema-v3 dataset
  validation remains covered. The locked direct-injection transition behavior
  and `saved_action_reproduces_lpf_transition=False` were preserved.
- The new untouched final-test range was locked as seeds 8,000,000 through
  8,000,099. Demonstration commissioning used the disjoint range beginning at
  5,000,100. Production generation uses only the four disjoint seed ranges
  predeclared below.
- The retained-v2 prefix regression artifact is
  `logs/go2_barrel_roll_v4/gate_v42_retained_prefix/`. Saved actions, applied
  torques, `qpos`, and `qvel` have maximum absolute difference 0.0 through all
  first 70 control steps. The candidate trajectory SHA-256 is
  `6882e4a638d20e42101b1dd2f87fa65f672ca76ad41b7e5aa8c42f5b2886f9ea`.
  This is a retained-prefix replay result, not a claim of independent-solver
  bit determinism.
- An initial commissioning run is preserved under
  `logs/go2_barrel_roll_v4/demonstrations/`. It was stopped after a rejected
  solver attempt without renderable frames exposed an error-reporting defect;
  that defect could mask the underlying rejection with a rendering error. The
  generator now preserves the solver rejection and renders only accepted saved
  control-endpoint states. None of the partial run was promoted or overwritten.
- The review set is
  `logs/go2_barrel_roll_v4/demonstrations_attempt2/`. It accepted 10 of 12
  attempts (seeds 5,000,100 through 5,000,106 and 5,000,109 through
  5,000,111). Seeds 5,000,107 and 5,000,108 were correctly rejected for
  immediate non-foot ground contact. The accepted files contain 1,250 total
  transitions and 39,756,043 bytes. All 10 pass strict schema-v4 validation,
  all checksum-index entries verify, and all accepted non-foot-contact counts
  are zero.
- The review videos are H.264, 640 by 480, 50 fps, 2.50 seconds, and exactly
  125 frames each. Mean/max action-clipping fractions are 0.003267/0.006667;
  mean MPX and applied-torque saturation fractions are 0.011100 and 0.010383.
  Generation completed in 3:33.32 wall time with 3,222,108 KiB maximum resident
  memory and no swap.
- Review evidence hashes are: effective configuration
  `34e0644dfbc2b42ef800ed00e12d9fc4d24ee82825aaddbf7a970c6792953a66`,
  checksum index
  `4890fe48399f5adcb21faf6bad736a74d1ca84628cc168f774e0fa1838c1212d`,
  generation manifest
  `8bf45b495effa2aa6df750672cfe38e941fb40b5d9e6a607cdc1ca993db2a81d`,
  dataset summary
  `bf16fdec8886a3e459259acfe3ddea6cc71ac1899ec30c7225d28d6abbcd0212`,
  and demonstration report
  `60403817703b44a27e5546efe3db257ea078419aae71201d09b3ed2e775e1e4b`.
  The retained schema-v2 and schema-v3 checksum-index hashes remain
  `153334c544fec09e8aee4fa74223bde0f5b1c6b4f0018ce1e65328fbf0ccee70`
  and
  `3c4401e353b3195e7c8801ae29257e84598f099c3bfab05b37dcb1c80c62c2cc`.
- Focused validation passed with 68 tests covering the controller, task,
  dataset, final evaluator, and injection path, plus 22 filtered environment,
  architecture, and TensorBoard tests. The only emitted diagnostics were two
  third-party deprecation warnings. Python byte-compilation and
  `git diff --check` also passed. The approval recorded above closed that
  mandatory visual-review pause.

### Gate V4.3 predeclared production execution

The production launch is locked to four workers, each requiring exactly 500
accepted trajectories:

| Worker | Start seed | Maximum attempts | Inclusive allowed range |
| --- | ---: | ---: | ---: |
| 0 | 6,000,000 | 5,000 | 6,000,000--6,004,999 |
| 1 | 6,100,000 | 5,000 | 6,100,000--6,104,999 |
| 2 | 6,200,000 | 5,000 | 6,200,000--6,204,999 |
| 3 | 6,300,000 | 5,000 | 6,300,000--6,304,999 |

These ranges are pairwise disjoint. They do not intersect the retained-v2
seeds (10,002--310,360), retained-v3 seeds (4,000,000--4,300,303), v4
commissioning and demonstration seeds (5,000,000--5,000,111), fixed validation
seeds (2,000,000--2,000,099), superseded final seeds
(3,000,000--3,000,099), or sealed v4 final seeds
(8,000,000--8,000,099). The gaps between worker ranges are intentional and
prevent one worker's maximum attempt budget from entering another worker's
range.

Worker evidence will be written under
`logs/go2_barrel_roll_v4/production/workers/`. The reconciled candidate will be
built at `data/go2_barrel_roll/v4.staging_production` and promoted by one
same-filesystem rename to `data/go2_barrel_roll/v4/` only after every production
check passes. All paths are refusal-to-overwrite. No commissioning traces or
videos are generated for the production set.

### Gate V4.3 production result

Gate V4.3 passed. All four workers exited zero after accepting exactly 500
trajectories. Their attempt counts were 1,181, 1,191, 1,147, and 1,179,
respectively. The combined manifests contain 4,698 unique attempted seeds,
2,000 unique accepted seeds, and 2,698 preserved rejections. Rejection counts
are 2,059 non-foot ground contacts, 317 excessive base linear speeds, 218
non-finite solver outputs, 49 insufficient-foot-support outcomes, 42
out-of-band rotations, 11 excessive body tilts, and 2 excessive joint speeds.

The promoted dataset contains exactly 2,000 schema-v4 archives, 250,000 direct
transitions, and 8,108,547,376 trajectory bytes. Strict aggregation validated
every archive. An independent `sha256sum -c --quiet checksums.sha256` pass
verified all 2,000 entries. Staging was removed by the same-filesystem rename,
and `logs/go2_barrel_roll_v4/production/PROMOTED.json` binds the promoted
production report. The dataset effective-configuration hash is
`34e0644dfbc2b42ef800ed00e12d9fc4d24ee82825aaddbf7a970c6792953a66`.

Production evidence hashes are:

- checksum index:
  `f9a85e1465a5d8b21f64c1ac7b47701de56a8c22c1b9741affa6a7e6cd45a156`;
- aggregate manifest:
  `bd0b88b2f42bd03b26fb9ee81903b6a5d9fb687dedf5143b22438535b9211171`;
- dataset summary:
  `186955b9271b9b58b07dad5c16e2b58e5c4fa0a25ff7d56209e66f589bef38f4`;
- production report:
  `b8dc60b00a09066b8e524e2a1d6e21475c906737817004b3fca9e2aa775bb7e2`;
- dataset completion marker:
  `5c6ad5db1c2736b504936c389c48b3d823619902df0a990598c9b2dea5056ae8`;
  and
- promotion marker:
  `85214286761210259db3b1f69b48bf2343be9b0eb915535dbb018229cba43a58`.

The retained schema-v2 and schema-v3 checksum-index hashes remain
`153334c544fec09e8aee4fa74223bde0f5b1c6b4f0018ce1e65328fbf0ccee70`
and
`3c4401e353b3195e7c8801ae29257e84598f099c3bfab05b37dcb1c80c62c2cc`.
All workers reported zero swap and no process failure. Total production wall
time was 8,245.49 seconds. Maximum worker resident memory was 2,845,780 KiB;
minimum observed host-available memory was 42,545,284 KiB. The system-wide GPU
monitor recorded a 28,877 MiB peak while validation suites ran concurrently;
the four generator processes alone were observed at approximately 4.0 GiB
after those suites exited, so the report preserves the higher conservative
system-wide peak rather than attributing it solely to generation.

The promoted directory passes the trainer's production-provenance gate,
including 16 exact runtime-source hashes and the COMPLETE-marker binding.
Direct-injection, task, data, evaluator, architecture, and retained-behavior
validation passed with 185 tests and one intentional skip. Python
byte-compilation, checksum verification, and `git diff --check` also passed.
The previously stale architecture and velocity-reward tests now assert the
unchanged SB3 256x256 default and the reward values already present in HEAD;
no model architecture or reward behavior was changed to make them pass.

### Gate V4.4 fixed preflight

The refusal-to-write preflight passed at
`2026-08-09T00:24:20.703289+00:00`. The campaign directory did not exist and
567,353,835,520 bytes were free. It locked concurrent training seeds 1 and 2
to exactly 500,000 environment steps each, four environments, 125-step
episodes, 25% random-selection direct MPC injection, 10,000-step validation,
25,000-step model and normalization checkpoints, disabled domain
randomization, enabled Go2 sysID, and no replay-buffer artifacts. No 0%
baseline command is present.

The preflight bound the promoted dataset hashes recorded above and the
following execution-source hashes:

- campaign launcher:
  `d071949a6beee81c3be1eccf81cfb0f951f43884787f6b3d3a9b8a603f10d7b8`;
- evaluator:
  `dcd389b7170dd8119a6a8ae00cf5741368c97e4d74ccbfe609269e1c6a7268bd`;
  and
- trainer:
  `7c157506978516af03b5773be0ecd04a7bdcb8bfce35f8697109f273a063c978`.

The launcher will refuse an existing campaign directory and will recheck
these hashes after both training processes finish before it permits selected-
policy validation review. Each run independently records and rechecks the 16
runtime-source hashes in its own configuration and completion marker. The
sealed final-test seeds remain unevaluated.

The campaign launch created
`logs/go2_barrel_roll_v4/campaign/campaign_plan.json` and started seed-1 and
seed-2 `/usr/bin/time` wrappers as PIDs 2,603,356 and 2,603,357, with trainer
children 2,603,358 and 2,603,360. Both generated run configurations were
audited before step-zero evaluation. They contain the exact locked options,
51 validation steps from 0 through 500,000, validation seeds 2,000,000 through
2,000,099, the sealed final range 8,000,000 through 8,000,099, the promoted
dataset hashes, and all 16 runtime-source hashes. Both trainer processes were
live; no evaluation-history record or selected model existed yet at that
initialization checkpoint.

### Gate V4.4 production training result

Gate V4.4 passed. Both concurrent trainers exited zero after exactly 500,000
steps. Each run contains the exact 51 evaluation records for steps 0 through
500,000, 20 model checkpoints, 20 normalization checkpoints, a TensorBoard
event, final-model artifacts, a write-once completion marker, and no replay
buffer artifact. The campaign rechecked the locked launcher, trainer,
evaluator, dataset, and 16 runtime-source hashes after training.

The locked checkpoint order selected:

- training seed 1 at step 300,000: 48/100 strict validation successes and
  mean final-hold standing score 0.7406816143064604; and
- training seed 2 at step 90,000: 31/100 strict validation successes and mean
  final-hold standing score 0.5973497754122536.

Later training was not monotonically better. Seed 1's 500,000-step policy
scored 12/100 and seed 2's scored 0/100; the preserved best-model directories
therefore correctly retain the earlier winners. Selected-model SHA-256 hashes
are `47aee5873bb28af2e707c68081216feb2bcb87b4613d883f981834eb5f17c9d8`
and `6635962b3ef26ec2517938ef67cf5cf24af41bed81afeff9ebbb9c3d5bf32e53`.
Selected normalization hashes are
`1c01a013b5d8f3db94e974f9cb568c50eb8fbf8c64f410bdb7edc0c76eb12de6`
and `ef741caa8ad5a53958010db6c285429b2c66e63fe1486f5e9b6e93f844c7ab24`.

Seed-1 and seed-2 wall times were 1:41:26 and 1:41:21. Maximum resident
memory was 3,142,696 KiB and 3,142,136 KiB, with zero swap. The campaign
monitor recorded 6,087.84 seconds wall time, a 2,367 MiB GPU-memory peak, and
49,828,392 KiB minimum host-available memory. Campaign validation and
completion-marker hashes are
`c8062cf5f1060531a6df2327d386e96f7cfd1898c3b5864105ac4c1c566af579`
and `7b6417c6efcd0e2a4eb139f3193934512f3fa901f7743bb95391aa8e61bd58d8`.

### Gate V4.5 selected-policy review result

The selected-policy reruns exactly matched their recorded 100-seed validation
results. The review package at
`logs/go2_barrel_roll_v4/campaign/validation_review/` contains 10 deterministic
success videos per selected policy, as explicitly requested by the user,
plus representative failures. Seed 1 has three representative failures:
insufficient foot support, excessive base linear speed, and excessive base
angular speed. Seed 2 has five: those three plus excessive joint speed and
non-foot ground contact.

The launcher's first automatic review attempt failed before rendering because
the evaluator passed diagnostic failure-reason strings to a recorder whose
filename-label contract permits only `success` and `failure`. The failure is
preserved in `logs/go2_barrel_roll_v4/campaign/failure.json`, and its partial
review is preserved under
`logs/go2_barrel_roll_v4/campaign/validation_review_failed_20260809T020657Z/`.
No training-frozen source was edited. A review-only adapter maps every
diagnostic failure label to `failure` while retaining the detailed reason in
the report; its source hash is
`34b9dd66f6d6ec86852bb8b242518bc4b995214ba7e726b8ce66e51e6d70f10c`.
The adapter and existing evaluator tests passed 8/8 before the clean rerun.

The clean rerun exited zero in 7:18.57, used 2,398,944 KiB maximum resident
memory, and used no swap. An independent audit verified all 28 report-bound
videos: every SHA-256 matches, every stream is H.264 at 640 by 480 and 50 fps,
and the two success-video counts are exactly 10. The review report,
ready marker, and adapter-provenance hashes are
`06a17a84930859c1563e116e9ca760d51de5ccbeca673c6be89b7adadf754f5d`,
`00f9f75831c99e20e381f3beff85324dae20f1ccd5c566bccd80a5a02a5213ce`,
and `3486ac36ac4c811f9995b96bd88a2b2eed5386235ed41ae2b2382df82cc4be6e`.
The repaired campaign-execution report hash is
`591944ca5fe24444396f85c4be09e5aa3f395224e67184f1e4721292b2db44ce`.
Post-review regression validation passed with 187 tests and one intentional
skip; Python byte-compilation, JSON parsing, and `git diff --check` also passed.
The four warnings are the same third-party deprecation and float32-cast
warnings already documented by the suite.
No validation approval marker or final-evaluation directory exists. Gate V4.5
is paused for the user's visual decision.

## Evidence motivating the recovery

- The selected schema-v2 policies usually execute a genuine airborne lateral
  roll and reach a supported stance, but the official 1.40-second horizon does
  not train or score the later behavior. Extended videos show later limb
  chatter, falling, or another maneuver.
- Schema v3 intentionally removed non-foot-contact failure and used a lenient
  final-only classifier. Its selected policy can fall onto the body and rotate
  on the ground while still passing.
- The schema-v3 MPC controller itself is not the primary failure. Its nominal
  trajectory executes the desired roll and terminal stance. The task reward,
  success predicate, dataset acceptance, and checkpoint metric admitted the
  wrong learned behavior.
- The 1,000 existing schema-v3 files cannot be reused unchanged: 401 record at
  least one non-foot ground contact, and only 488 satisfy all schema-v4 final
  hold conditions when audited at the 50 Hz control endpoints.
- Schema-v2 MPC terminal behavior at 1.40 seconds supports the selected
  standing scales:

  | Condition | Schema-v2 trajectories passing at 1.40 s |
  | --- | ---: |
  | Four-foot support | 100.0% |
  | Base height at least 0.20 m | 100.0% |
  | Body tilt at most 0.35 rad | 99.9% |
  | Base linear speed at most 0.10 m/s | 70.1% |
  | Base angular speed at most 0.50 rad/s | 98.3% |
  | Joint-velocity norm at most 1.0 rad/s | 83.8% |

  The corresponding medians are 0.272 m, 0.143 rad, 0.0717 m/s,
  0.150 rad/s, and 0.560 rad/s. The standing reward begins at 1.40 seconds and
  is smooth, so a trajectory that is still settling receives partial reward.
  The strict hold does not begin until 2.00 seconds.

## Locked task contract

### Timing and controller phases

- Simulation timestep: 0.005 seconds (200 Hz).
- Policy control timestep: 0.02 seconds (50 Hz).
- Episode horizon: 2.50 seconds, 125 control transitions and 500 physics
  transitions.
- Preserve the schema-v2 desired-roll and MPC maneuver schedule through
  1.40 seconds.
- The desired roll reaches one positive turn, `2*pi`, at 0.80 seconds and
  remains there.
- The actor phase scalar uses schema-v2 semantics: it reaches 1.0 at
  0.80 seconds and remains constant for the rest of the episode.
- After 1.40 seconds, stop MPC replanning and repeat the terminal-padded final
  stance plan with the same PD feedback through 2.50 seconds.
- Do not add an explicit flight-duration success requirement.

### Immediate failures

Terminate immediately with terminal reward -25 for:

1. a non-finite physics state; or
2. any robot non-foot geometry contacting the ground.

Non-foot contact is evaluated at every physics substep and is never filtered.
One- or two-foot first touchdown remains valid.

### Final standing hold

The last 0.50 seconds contain exactly 25 control transitions. A successful
episode must satisfy every condition below at every one of those 25 control
endpoints:

- all four feet have filtered ground contact;
- directed unwrapped roll progress is within `2*pi +/- 0.35` rad;
- base height is at least 0.20 m;
- body-up/world-up tilt is at most 0.35 rad;
- base linear-speed norm is less than 0.10 m/s;
- base angular-speed norm is less than 0.50 rad/s; and
- joint-velocity norm is less than 1.0 rad/s.

Implement foot-contact filtering as
`filtered_contact[t] = raw_contact[t] OR raw_contact[t-1]`, using a genuinely
previous control endpoint rather than a value already overwritten by the base
environment. This tolerates one 20 ms contact-reporting dropout. A second
consecutive dropout fails the four-foot condition. The hold-valid streak
resets whenever any condition fails. Terminal success requires a streak of at
least 25 at control step 125.

Except for the two immediate failures, do not terminate early. Near misses
must be allowed to continue so SAC can learn recovery behavior.

Use deterministic failure precedence and log each failed condition separately.
At minimum preserve specific categories for non-finite state, non-foot contact,
rotation, foot support, height, tilt, base linear speed, base angular speed,
and joint speed.

## Locked reward contract

Retain the schema-v2 flip reward and add one post-maneuver standing score. Do
not retain schema v3's desired-angular-rate term. Do not add an action-change
penalty; the existing 5 Hz action low-pass filter remains active.

Let:

```text
roll_tracking = exp(-((roll_progress - desired_roll) / 0.50)^2)

expected_roll_step = 2*pi*control_dt / (roll_end_time - roll_start_time)
signed_progress = 0.25 * clip(
    direction * (roll_progress - previous_roll_progress) / expected_roll_step,
    -3.0,
    3.0,
)
```

`roll_tracking` is active for the complete 2.50-second episode.
`signed_progress` is active only through 0.80 seconds and is exactly zero
afterward. This removes the positive incentive for a second rotation while
continuing to penalize movement away from one unwrapped turn.

The standing score becomes active at 1.40 seconds. Define:

```text
foot_score = mean(filtered_foot_contacts)
height_score = exp(-((base_height - 0.27) / (0.27 - 0.20))^2)
tilt_score = exp(-(body_up_tilt / 0.35)^2)
linear_speed_score = exp(-(base_linear_speed / 0.10)^2)
angular_speed_score = exp(-(base_angular_speed / 0.50)^2)
joint_speed_score = exp(-(joint_velocity_norm / 1.0)^2)

standing_score = mean(
    foot_score,
    height_score,
    tilt_score,
    linear_speed_score,
    angular_speed_score,
    joint_speed_score,
)
```

Before 1.40 seconds, `standing_score` is zero. The per-step reward is:

```text
reward = roll_tracking + signed_progress + standing_score + terminal_outcome
```

`terminal_outcome` is +25 for terminal success, -25 for any terminal failure,
and zero otherwise. Log every component and every unweighted standing
sub-score separately. Check reward magnitudes over accepted MPC trajectories
before training; do not change weights merely to make a learning run pass.

## Observation, action, and learning contracts

- Keep the 45-dimensional actor observation layout unchanged except for
  restoring the schema-v2 saturated phase semantics.
- Keep the 4-dimensional privileged critic observation unchanged.
- Keep the 12-dimensional position-residual action interface,
  `action_scale=2.0`, 5 Hz LPF, PD gains, torque limits, and Go2 system
  identification unchanged.
- Keep SAC architecture, optimizer, hyperparameters, four training
  environments, disabled domain randomization, and the existing 25%
  percentage-injection algorithm unchanged.
- Keep random trajectory selection unchanged. Do not change it to a shuffled
  cycle or attempt to force every dataset file into a replay buffer.
- Preserve direct transition injection, terminal/timeout semantics, source
  tagging, and strict schema routing.
- Preserve `saved_action_reproduces_lpf_transition=False`. The known
  inverse-PD-label/direct-torque-transition mismatch is an intentional retained
  risk for this recovery because schema v2 demonstrated that the existing
  injection pipeline can learn the desired flip. Investigate it only after a
  failed schema-v4 campaign provides evidence that it owns the failure.

## Schema-v4 dataset contract

Create `data/go2_barrel_roll/v4/` without modifying or overwriting v2 or v3.
Schema-v4 training must reject v1, v2, and v3 files.

Each accepted file must contain the complete 125-transition episode and enough
state, contact, action, reward-component, classifier, controller, timing, and
provenance data to recompute all observations, rewards, failures, and final
hold conditions. Strict validation must reject:

- wrong shapes, timing, schema, task, robot, direction, action semantics, LPF,
  gains, or configuration hashes;
- non-finite arrays;
- inconsistent observations, roll progress, rewards, done flags, or metadata;
- any non-foot ground contact;
- fewer than 25 consecutive valid final-hold control endpoints;
- early termination in an accepted trajectory; or
- any checksum, manifest, seed, or filename inconsistency.

Only accepted successes become injectable `.npz` files. Every rejected attempt
remains in worker and aggregate manifests with its exact failure reason.

### Demonstration approval gate

1. Generate 10 accepted varied schema-v4 MPC trajectories.
2. Strictly validate every file.
3. Render all 10 complete 2.50-second trajectories.
4. Report per-step hold metrics, non-foot-contact evidence, reward components,
   action clipping, torque saturation, and source identities.
5. Pause for the user to inspect the videos. Do not start production data
   generation until the user explicitly approves them.

### Production data gate

After user approval, generate exactly 2,000 unique accepted trajectories. Use
four isolated workers with 500 accepted trajectories each and disjoint,
predeclared seed ranges. The implementation agent must record the exact starts
and maximum-attempt limits before launch and prove no overlap with calibration,
validation, or final-test seeds.

Before atomic promotion from staging:

1. strictly validate all 2,000 files;
2. prove exactly 250,000 direct transitions;
3. reconcile files and seeds against all worker manifests;
4. aggregate and report every rejection category;
5. create and verify a checksum index and effective-configuration hash;
6. report trajectory bytes, wall time, GPU memory, host memory, and process
   failures; and
7. prove the retained v2 and v3 checksum indexes are unchanged.

The estimated schema-v4 trajectory storage is approximately 8.3 GB. This is a
planning estimate, not an acceptance result.

## Implementation and regression gates

### Gate V4.1: shared task semantics

Implement versioned constants, reward functions, hold classification,
termination, info fields, TensorBoard fields, and schema-v4 observation phase.
Preserve v1-v3 validation behavior for historical files.

Focused tests must cover:

- exact 2.50-second timing and 25-step final hold;
- phase saturation at 0.80 seconds;
- signed-progress gating after 0.80 seconds;
- standing-score gating at 1.40 seconds;
- roll tracking through 2.50 seconds;
- every smooth standing component and terminal +/-25 outcome;
- immediate non-finite and non-foot-contact termination;
- genuinely previous-step contact filtering;
- every strict threshold on, just inside, and just outside its boundary;
- streak reset after any invalid final-hold step;
- rejection of a second rotation; and
- no explicit flight-duration requirement.

### Gate V4.2: generator and retained-v2 regression

Restore the v2 generator/controller behavior for the first 1.40 seconds and
append only the terminal-padded standing tail. Use the retained schema-v2
regression artifact to prove that the task plant, saved actions, applied
torques, `qpos`, and `qvel` through the first 70 control steps have not changed
when replaying the retained prefix. Independently commission fresh MPC solves;
do not claim bit determinism across independent GPU solver processes.

Then complete the 10-video demonstration approval gate.

### Gate V4.3: dataset generation and injection validation

Complete the 2,000-file production data gate. Add strict direct-injection tests
that load schema v4, preserve exact stored tuples after the replay buffer's
float32 conversion, maintain terminal timeout semantics, reject older schemas,
and reach the requested 25% composition without changing selection behavior.

### Gate V4.4: concurrent production training

Train seeds 1 and 2 concurrently for exactly 500,000 environment steps each.
Use separate, refusal-to-overwrite run directories. Save model and
normalization checkpoints, complete provenance, TensorBoard events, evaluation
history, failure diagnostics, and resource logs. Do not save replay buffers
unless a new instruction explicitly requests them.

Use fixed validation seeds 2,000,000 through 2,000,099 at step zero and every
10,000 training steps. Training may pause synchronously for these evaluations
and then continue. Each evaluation must report strict success, all failure
reasons, final-hold streaks, standing sub-scores, motion distributions, reward
components, roll progress, contacts, and returns.

For each training seed, select the checkpoint by:

1. highest strict validation success rate;
2. highest mean final-hold standing score;
3. earliest training step.

Do not substitute `final_model.zip` unless it independently wins this order.

### Gate V4.5: validation video review

Only after both 500,000-step jobs finish:

1. re-evaluate each selected checkpoint on all 100 validation seeds;
2. verify the rerun matches the recorded selection result;
3. render 10 validation successes for each selected policy when available;
4. render representative failures and report why they failed; and
5. pause for user review.

Do not open the untouched final-test set until the user explicitly approves
the validation videos. The user, not the implementation agent, is the visual
authority.

### Gate V4.6: untouched final evaluation

Before training begins, lock a new 100-seed final-test range that is disjoint
from all prior generation, validation, and final-test seeds. Do not reuse
3,000,000 through 3,000,099. Keep the range sealed until Gate V4.5 passes.

After validation-video approval:

1. lock both selected checkpoints and all relevant hashes;
2. evaluate each checkpoint exactly once on the new 100 final seeds;
3. report both policies without discarding the weaker result;
4. randomly select and render 10 classifier-success final episodes for the
   user to inspect; and
5. render representative final failures.

Schema v4 passes only if at least one policy scores at least 80/100 under the
strict contract and the user approves its final-success videos. A numerical
pass without visual approval is not acceptance.

If both policies pass, choose the official artifact by final success rate,
then validation success rate, then mean final-hold standing score, then lower
training seed.

### Gate V4.7: documentation and final audit

Update the authoritative plan/status, README, result report, and
reproducibility documentation so they do not present schema-v3 classifier
success as satisfactory barrel-roll behavior. Preserve the historical results
and label them as superseded under the schema-v4 behavioral contract.

The final audit must include relevant focused and broader tests, Python
byte-compilation, shell syntax checks, `git diff --check`, strict validation of
all schema-v4 files and checksums, provenance reconciliation, artifact hashes,
video technical validation, and an explicit list of any failing checks.

Do not claim completion unless all automated gates pass, at least one policy
reaches 80/100 on the new untouched final test, and the user approves the
videos.

## Failure boundary and fallback

If either the MPC demonstration videos or policy videos are rejected, or if
neither policy reaches 80/100, stop and preserve all evidence. Do not
automatically launch more seeds, change thresholds, tune reward weights, add
observations, change architecture, alter SAC, change injection percentage, or
modify the action interface.

Produce a diagnostic comparison covering:

- early non-foot contacts;
- roll progress and second-rotation attempts;
- per-condition final-hold failures and streak lengths;
- height, tilt, base motion, joint motion, and contact distributions;
- reward and standing sub-score distributions;
- raw action changes despite the retained LPF;
- replay composition and sampled trajectory identities;
- critic, value, entropy, and loss behavior; and
- representative videos.

Use that evidence to identify the earliest owning contract or pipeline gate
before proposing another recovery.

## Explicit non-goals

- No 0% MPC-injection baseline until schema-v4 25% training succeeds.
- No controller handoff after landing.
- No explicit flight-duration classifier.
- No action-change reward penalty in the first schema-v4 campaign.
- No actor/critic architecture or observation-dimension change.
- No action-interface, LPF, PD, torque, SAC, or percentage-injection change.
- No domain randomization, seed sweep, reward sweep, or automatic schema-v5
  fallback.
- No real-robot deployment or hardware-validation claim.
