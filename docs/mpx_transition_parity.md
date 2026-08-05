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
