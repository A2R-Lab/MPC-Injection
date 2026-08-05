# MPX direct-transition parity contract

This contract freezes the first-milestone comparison before measuring the
current MPX inferred-action/direct-torque path. The machine-readable source of
truth is
`docs/mpx_bound_milestone_reports/transition_parity_tolerances.json`.

## Compared transitions

The source transition is produced by the canonical
`mpc_rl/planner/gen_traj_data_mpx_dr.py` path with actual MPX output. The replay
transition starts from the same reset state and startup-DR patch, then calls
`QuadrupedVelocityTrackingEnv.step(saved_action)`. No policy, fake controller,
or torque replay substitutes for this comparison.

The fixed scenarios use seed 17 for 160 control steps (640 simulation
substeps): nominal dynamics (`disabled`) and the deterministic sampled
`sysid_dyn20_mjlab` patch. The DR replay must reproduce the same seeded push
schedule and at least one nonzero push. Observations are clean in both source
and replay, matching the canonical generator.

For control step 0 and for the full sequential rollout, the report contains
maximum absolute and RMS disagreement for post-step `qpos`, post-step `qvel`,
policy observation, privileged observation, reward, applied substep torque,
and push velocity delta. Termination, truncation, action range, push-event
locations, action clipping, and torque-saturation masks are compared exactly
or by the separately declared mask limits. Metrics are aggregated over every
element; the report also retains per-control-step maxima so drift is visible.

## Frozen tolerances

The maximum/RMS limits are respectively:

- `qpos`: `1e-6` / `1e-7`;
- `qvel`: `1e-5` / `1e-6`;
- policy and privileged observations: `1e-5` / `1e-6`;
- reward: `2e-6` / `5e-7` (the generator stores rewards as float32);
- applied substep torque: `1e-5` / `1e-6`;
- deterministic push delta: `1e-12` / `1e-13`.

Termination, truncation, saved-action range, and push-event mismatches must all
be zero. Torque-saturation mask disagreement may cover at most 0.1% of joint
substeps to tolerate a value landing numerically on a limit; replay action
clipping must be zero. The source inverse-PD conversion may clip at most 1% of
action elements, with maximum raw overshoot `0.05`.

These thresholds are intentionally far above float64 deterministic
repeatability while remaining well below the previously observed fake-MPC
disagreement. A path passes only if every one-step and full-rollout criterion
passes in both scenarios. If it fails, the report is retained and the
environment-step mapping from the bounding-data plan is evaluated under this
same declaration ID; the thresholds are not relaxed.

## Push-coverage amendment

The retained v1 baseline report showed that the selected
`sysid_dyn20_mjlab` preset samples push intervals from 5–10 seconds, while the
predeclared short rollout is 3.2 seconds. Thus v1 could not exercise a push and
failed its explicit push-coverage requirement. This setup error does not erase
the measured transition failures.

`transition_parity_tolerances_v2.json` corrects only the coverage apparatus:
it predeclares fixed validation-only velocity deltas after zero-based control
steps 39 and 99. All numeric, exact, clipping, saturation, seed, and horizon
tolerances are unchanged from v1. The unchanged direct path must be measured
again under v2 before the conditional environment-step mapping is evaluated
under the same v2 contract.

The conditional mapping evaluation uses
`transition_parity_env_step_tolerances_v2.json`. It changes only the declared
conversion mode to `env_step_lpf_inverse_v1`; every scenario, seed, horizon,
push, numeric, exact, clipping, and saturation threshold is identical to v2.

## Measured decision

The v2 actual-MPX direct-torque report failed. Nominal one-step maximum
disagreement was `0.0600846` for qpos and `4.24142` for qvel; nominal
full-rollout maxima were `0.180062` and `6.22611`. Under DR, the source fell
after 127 control steps and full-rollout maxima reached `0.991056` for qpos and
`10.9049` for qvel. The deterministic pushes occurred at steps 39 and 99 with
no event mismatch. Source action conversion also exceeded its limits:
3.95833% clipped elements and `1.20508` maximum overshoot nominally, and
16.4698% / `2.96550` under DR. The retained machine-readable evidence is
`transition_parity_baseline_report_v2.json`.

The conditional environment-step source then completed both 160-step actual-
MPX rollouts without a fall and replayed all state, observation, applied-
torque, discrete, saturation-mask, and deterministic-push fields within the
frozen tolerances. State, observation, torque, and push disagreements were
exactly zero; reward max/RMS disagreement was at most `5.80609e-8` /
`3.39942e-8`. It still failed the unchanged action-conversion limits: nominal
clipping was 13.3854% with `3.18516` maximum overshoot, and DR clipping was
23.4375% with `4.81915` maximum overshoot, versus limits of 1% and `0.05`.
The exact report is `transition_parity_env_step_report_v2.json`.

Therefore neither conversion is qualified for accepted direct-transition
data. The legacy direct-torque implementation remains available under its
explicit mode for compatibility and diagnostics; the environment-step mapping
is retained as a separately named evaluated candidate, but it does not replace
the default. Per the gated plan, gait calibration, nominal bound acceptance,
and rendered candidate generation are blocked pending an explicit environment
action-interface decision. The frozen thresholds were not changed.
