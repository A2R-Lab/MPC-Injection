# MPX 0.5 m/s bounding controller: working configuration

Status: the controller completed the required 1,000 control-step (20 s)
nominal horizon at 0.5 m/s for seeds 1000 and 1001 after the controller fixes
below. No safety contact or termination occurred in either final run.

## Working configuration

- gait: `bound_front_first`
- target command: `[0.5, 0.0, 0.0]` m/s, m/s, rad/s
- command/gait transition: 50 control steps
- duty factor: `0.65`
- step frequency: `2.5 Hz`
- physical swing apex: `0.05 m`
- MPX pitch orientation cost: `25000`
- MPX pitch-rate cost: `1000`
- action interface: `mpx_bound_scale_1_no_lpf_v1`
- nominal dynamics with the Go2 sysID patch; startup domain randomization disabled

Reproduce one rollout from the repository root with:

```bash
PYTHONPATH="$PWD/deps/mpx:$PWD/deps/gym-quadruped:$PWD" \
conda run --no-capture-output -n mpc-rl \
python mpc_rl/planner/gen_traj_data_mpx_dr.py \
  --num-trajectories 1 \
  --max-attempts 1 \
  --episode-length 1000 \
  --start-seed 1000 \
  --domain-rand-config-type disabled \
  --target-command 0.5 0 0 \
  --command-ramp-control-steps 50 \
  --gait bound_front_first \
  --duty-factor 0.65 \
  --step-frequency-hz 2.5 \
  --step-height-m 0.05 \
  --mpx-qrot-pitch-cost 25000 \
  --mpx-qomega-pitch-cost 1000 \
  --action-interface mpx_bound_scale_1_no_lpf_v1
```

## Final measured results

| Metric | Seed 1000 | Seed 1001 |
| --- | ---: | ---: |
| Completed control steps | 1000 | 1000 |
| Mean hold forward velocity | 0.50143 m/s | 0.50065 m/s |
| Complete front/rear cycles | 47 | 47 |
| Paired-interval alternation | 1.000 | 1.000 |
| Front pair agreement | 0.99553 | 0.99658 |
| Rear pair agreement | 0.99605 | 0.99079 |
| Diagonal/lateral-only support | 0.000 | 0.000 |
| Action clipping fraction | 0.00025 | 0.00233 |
| Maximum normalized clip excess | 0.06809 | 0.39354 |
| Minimum base height | 0.23467 m | 0.21137 m |
| Safety contact / termination | none | none |

The user-requested goal is bounding behavior, so the second seed's maximum clip
excess is reported but is not treated as a reason to reject an otherwise safe,
stable bound.

## Controller fixes that made the gait work

1. Swing-foot spline derivatives now use physical m/s units. Horizontal
   derivatives are duration-scaled, and the vertical curve reaches the requested
   apex without overshoot while maintaining continuous liftoff/apex/landing
   velocity.
2. The command ramp now smoothly transitions duty factor from full stance and
   swing height from zero to their moving values, avoiding the abrupt first
   bound cycle.
3. Whole-body dynamics now solves the planned active-contact subsystem instead
   of solving a four-foot constraint system and masking swing-foot forces after
   the solve.
4. The wrapper rejects nonfinite, initial-state-inconsistent, or implausible
   solver output. It uses measured current joint positions for the MPX action
   mapping instead of trusting solver `X[0]` as a future joint reference.

## Remaining scope

This is nominal simulation evidence, not a domain-randomized robustness or
real-robot deployment claim. The next data-generation stage should retain this
controller unchanged and measure acceptance across independent reset and DR
seeds.

## Ten-seed video evaluation

A later exact-controller evaluation recorded seeds 1000–1009 at 640×480 and
50 FPS. Six seeds completed the full horizon; four recordings retain the first
detected failure for diagnosis. The video index and exact JSON metrics are in
`data/quadruped_bound/nominal_vx0p5/video_evaluation_working_v1_seeds1000_1009/`.

The reproducible recorder is
`mpc_rl/planner/evaluate_mpx_bound_videos.py`. It streams frames directly to
`ffmpeg`, so full rollouts are not buffered in memory.
