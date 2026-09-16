# MPX 0.5 m/s bounding commissioning: extended cost diagnostics

Status: no automatic pass; freeze, render, pilot, and production remain gated.
This report extends `vx0p5_commissioning_exhausted_v1.md`. Exact source records
remain in each ignored local `generation_manifest.jsonl` below; bulk artifacts
are not committed.

## Action decomposition

An unchanged 250-step diagnostic of the strongest configuration decomposed the
raw action as `q_des - q_default + tau_ff / kp`. At the worst element (RR thigh,
control step 215), the normalized raw action was 1.786926:

- MPX position-reference residual: 0.663796 rad;
- feedforward torque: 22.462595 Nm;
- feedforward contribution at nominal thigh `Kp=20`: 1.123130.

All four thighs reached the MPX 25 Nm output cap during the trajectory, so the
feedforward contribution alone could reach 1.25 normalized action. This evidence
motivated the torque, joint-reference, joint-velocity, and matched-gain trials
below. Actuator limits and the accepted action interface were not changed.

## Additional retained attempts

| Artifact directory suffix | Result |
| --- | --- |
| `front_freq3_h0p03_qpitch25k_qomega1000_qpz2500` | RR calf contact, step 158 |
| `front_freq3_h0p03_qpitch25k_qomega1000_qpz5000` | Completed; 9 cycles, 0.692308 alternation, 0.033000 clipped fraction, 1576257.925 max excess |
| `front_freq3_h0p03_qpitch25k_qomega1000_qpz20000` | RR thigh contact, step 102 |
| `front_freq3_h0p03_qpitch25k_qomega1000_qdpz1000` | Torso contact, step 150 |
| `front_freq3_h0p03_qpitch25k_qomega1000_qdpz4000` | RL calf contact, step 174 |
| `front_freq3_h0p03_qpitch25k_qomega1000_qlegz50000` | FR calf contact, step 141 |
| `front_freq3_h0p03_qpitch25k_qomega1000_qlegz200000` | Completed; 10 cycles, 0.740741 alternation, 0.027667 clipped fraction, 0.795728 max excess |
| `front_freq3_h0p03_qpitch25k_qomega1000_swing0p2` | Completed; 5 cycles, 0.476190 alternation, 0.018333 clipped fraction, 0.573204 max excess |
| `front_freq3_h0p03_qpitch25k_qomega1000_swing0p6` | Completed; 7 cycles, 0.700000 alternation, 0.024667 clipped fraction, 0.954253 max excess |
| `front_freq3_h0p03_qpitch25k_qomega1000_qtau0p15` | RR thigh contact, step 137 |
| `front_freq3_h0p03_qpitch25k_qomega1000_qtau0p2` | RR calf contact, step 158 |
| `front_freq3_h0p03_qpitch25k_qomega1000_qtau1` | RR calf contact, step 26 |
| `front_freq3_h0p03_qpitch25k_qomega1000_qq1` | Completed; 7 cycles, 0.583333 alternation, 0.021333 clipped fraction, 0.655498 max excess |
| `front_freq3_h0p03_qpitch25k_qomega1000_qdq0p2` | RL calf contact, step 125 |
| `front_freq3_h0p03_qpitch25k_qomega1000_qdq1` | Torso contact, step 182 |
| `front_freq3_h0p03_qpitch25k_qomega1000_thighkp40kd2` | RL thigh contact, step 158 |

The tested parameter groups were body vertical-position and vertical-velocity
costs, vertical foot-reference cost, horizontal swing-foot initial speed,
torque effort, joint-position reference, joint-velocity damping, and matched
thigh feedback gains. Each group was changed independently from the retained
3 Hz, 0.03 m, 0.5-duty, 25,000 pitch-orientation, 1,000 pitch-rate candidate.

## Required-horizon result

`full20s_front_freq3_h0p03_qpitch25k_qomega1000` evaluated the strongest
unchanged candidate at the required 1,000-control-step horizon. It failed at
step 407 from an RL calf ground contact. Partial evidence before rejection was:

- clipped-element fraction 0.026003 and maximum excess 0.850831;
- front-only fraction 0.175070 and rear-only fraction 0.126050;
- front-pair agreement 0.852941 and rear-pair agreement 0.950980;
- minimum base height 0.169726 m, maximum absolute pitch 0.144315 rad, and
  maximum absolute roll 0.072728 rad;
- all MPX outputs finite before the contact failure.

Thus, extending the horizon does not dilute the clipping failure and introduces
an unconditional safety rejection well before 1,000 steps.

## Decision boundary

No tested plan-authorized gait, phase, MPX reference/cost, or feedback-gain
configuration passes the fixed safety, measured-bound, and clipped-element
fraction gates. The plan's optional last-ditch change applies only to maximum
clip magnitude for an otherwise successful bound; it cannot cure the observed
1.8–3.3% clipped fractions, sub-0.8 alternation, or non-foot contacts.

There is still no configuration eligible for visual pass inspection or freeze.
Pilot and production generation must not start under the current evidence.
