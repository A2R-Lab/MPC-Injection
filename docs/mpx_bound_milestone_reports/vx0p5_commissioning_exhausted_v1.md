# MPX 0.5 m/s bounding commissioning: no passing candidate

Status: blocked before freeze. No 0.5 m/s commissioning trajectory passed the
predeclared automatic predicate, so no visual pass, frozen declaration, pilot,
or production generation was started.

The source-of-truth attempt records remain under the ignored local artifact
root `data/quadruped_bound/nominal_vx0p5/commissioning/`. Each directory listed
below contains its own `generation_manifest.jsonl`, `validation_summary.json`,
and checksum index. Those manifests preserve the exact declaration, realized
controller arrays, generator/submodule provenance, measured evidence, and
failure reasons. Bulk trajectory/manifests are intentionally not committed.

## Strongest completed candidates

| Candidate | Cycles | Alternation | Clipped elements | Max clip excess | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| `front_freq3_h0p03_qpitch25k_qomega1000` | 10 | 0.740741 | 0.019667 | 0.611639 | Failed alternation and both clipping gates |
| `front_freq3p2_h0p03_qpitch25k_qomega1000` | 8 | 0.739130 | 0.014667 | 0.478975 | Failed cycle, alternation, and both clipping gates |
| `front_freq3_h0p03_qpitch25k` | 9 | 0.642857 | 0.022000 | 0.827431 | Failed cycle, alternation, and both clipping gates |
| `front_freq3_h0p03_qpitch10k` | 6 | 0.565217 | 0.014667 | 0.551190 | Failed cycle, alternation, and both clipping gates |
| `front_freq3_h0p03_qpitch25k_qomega1000_thighkd0p5` | 8 | 0.571429 | 0.019000 | 0.696710 | Failed cycle, alternation, and both clipping gates |

All five completed candidates above passed the declared pair-agreement,
front/rear-support, diagonal/lateral-support, posture, and solver-finiteness
checks. None was eligible for full saved-action replay because static acceptance
failed first.

## Tuning inventory

The following retained directories cover the plan-permitted tuning order. A
plain “completed/rejected” entry means the full 250-step rollout was safe but
failed the measured gait and/or clipping predicate; safety failures give the
first reported contact and completed control-step count.

- Baseline/evidence: `retained_v1` (completed/rejected),
  `retained_v1_evidence` (RR thigh, step 191).
- Frequency/height: `freq3p5_h0p03` (RL thigh, 90), `freq3_h0p02`
  (FR calf, 142), `freq3_h0p045` (RL thigh, 158),
  `front_freq2p8_h0p03` (RL calf, 188), `front_freq3p2_h0p03`
  (completed/rejected).
- Duty/orientation: `front_freq3_h0p03_duty0p45` (completed/rejected),
  `front_freq3_h0p03_duty0p55` (RR calf, 209), `hind_freq3_h0p03`
  (FL calf, 47).
- Pitch orientation cost: `front_freq3_h0p03_qpitch10k`
  (completed/rejected), `front_freq3_h0p03_qpitch20k` (RL thigh, 207),
  `front_freq3_h0p03_qpitch25k` (completed/rejected).
- Body-height reference: `front_freq3_h0p03_qpitch25k_z0p24` (torso, 101),
  `front_freq3_h0p03_qpitch25k_z0p30` (completed/rejected).
- Pitch-rate cost: `front_freq3_h0p03_qpitch25k_qomega250`
  (completed/rejected), `front_freq3_h0p03_qpitch25k_qomega1000`
  (completed/rejected), `front_freq3_h0p03_qpitch25k_qomega1250`
  (RR thigh, 24), `front_freq3_h0p03_qpitch25k_qomega2000`
  (RR calf, 125).
- Feedback gains: `front_freq3_h0p03_qpitch25k_qomega1000_kp30`
  (RR calf, 141), `front_freq3_h0p03_qpitch25k_qomega1000_thighkp30`
  (RR calf, 126), `front_freq3_h0p03_qpitch25k_qomega1000_thighkd2`
  (torso, 82), and
  `front_freq3_h0p03_qpitch25k_qomega1000_thighkd0p5`
  (completed/rejected).
- Controlled combinations: `front_freq3p2_h0p03_qpitch25k_qomega1000`
  (completed/rejected), `front_freq3p1_h0p03_qpitch25k_qomega1000`
  (torso, 141), `front_freq3_h0p04_qpitch25k_qomega1000`
  (torso, 100), and `hind_freq3_h0p03_qpitch25k_qomega1000`
  (torso, 120).

The previously committed 0.1 m/s diagnostic remains separately recorded in
`canonical_vx0p1_regression_v1.json`; it is not a 0.5 m/s acceptance result.

## Seed replication of the strongest candidate

`replication_qpitch25k_qomega1000_seeds1001_1005` ran the unchanged strongest
candidate against five additional nominal-reset seeds. The environment keeps
seeded initial joint-position, roll/pitch, and joint-velocity perturbations even
when physics domain randomization is disabled.

| Seed | Completed steps | First safety failure | Partial clipped fraction | Partial max clip excess |
| ---: | ---: | --- | ---: | ---: |
| 1001 | 6 | FL calf | 0.000000 | 0.000000 |
| 1002 | 241 | RR calf | 0.018672 | 0.560731 |
| 1003 | 173 | RR calf | 0.024566 | 0.602829 |
| 1004 | 125 | RL calf | 0.025333 | 0.539105 |
| 1005 | 183 | FL calf | 0.027322 | 13644857.750000 |

This 0/5 safety result rules out freezing the strongest single-seed candidate.
The seed-1005 finite-but-extreme action is retained by the clipping evidence;
it was not hidden, clipped away from validation, or mistaken for an acceptable
controller output.

## Decision boundary

Commissioning exhausted the plan-authorized gait, sagittal MPX reference/cost,
phase-orientation, and joint-feedback surfaces without an automatic pass. The
plan permits a last-ditch revision only to maximum clipping magnitude after an
otherwise successful candidate; it does not authorize weakening the cycle,
alternation, clipped-element-fraction, or safety gates. The best candidate still
fails three of those immutable gates and fails 5/5 additional safety rollouts.

Proceeding requires a new user-approved scope or acceptance decision. Under the
current plan, freeze/pilot/production must remain blocked.
