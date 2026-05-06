Your updated stack changes the plan materially.

The key correction: in your deployment code, the 1 kHz loop is **not computing torques**. It is repeatedly publishing `LowCmd.q`, `dq`, `kp`, `kd`, and maybe `tau`; the actual PD tracking is inside Unitree firmware/motor control. Public Unitree SDK material describes low-level access through position, velocity, stiffness, damping, and feedforward torque fields, and the Go2 example writes `q`, `dq`, `kp`, `kd`, and `tau` into `motor_cmd()` before publishing `lowcmd`. ([DeepWiki][1])

So I would revise the plan as follows.

---

# Updated interpretation of your stack

Training:

```text
policy:      50 Hz
MuJoCo PD:   200 Hz
physics:     200 Hz
```

Deployment:

```text
policy:              50 Hz
LowCmd publishing:   1 kHz
actual motor PD:     Unitree internal, not directly set here
```

This means your training and deployment are both of the form:

```text
slow learned target update
fast low-level target tracking
```

That is good. It matches the “between-timestep” structure discussed in *Action Space Design*: when the learning/command frequency is lower than the control frequency, the same action persists across multiple lower-level updates while torque updates can occur multiple times using local state information. 

But the important mismatch is:

```text
training:   your code explicitly recomputes simulated PD torque at 200 Hz
deployment: Unitree firmware tracks q/dq/kp/kd internally at unknown motor-control rate
```

Your 1 kHz loop is best understood as a **command-refresh / target-conditioning loop**, not the actual torque loop.

---

# Revised priority order

## Priority 1: Keep the policy at 50 Hz and add target conditioning in the 1 kHz LowCmd loop

This is now the cleanest first intervention.

Instead of:

```cpp
lowcmd.motor_cmd()[i].q() = action[i];
```

do:

```cpp
q_policy[i] = latest 50 Hz ONNX output;
q_cmd[i]    = filtered / rate-limited / jerk-limited version of q_policy[i];

lowcmd.motor_cmd()[i].q()  = q_cmd[i];
lowcmd.motor_cmd()[i].dq() = dq_cmd[i];  // optional, but useful if generated consistently
lowcmd.motor_cmd()[i].kp() = kp[i];
lowcmd.motor_cmd()[i].kd() = kd[i];
```

Do **not** view this as lowering policy frequency. The ONNX policy still runs at 50 Hz. You are only changing the interpolation/conditioning between policy target updates.

This is the direct analogue of keeping (f_{\text{RL}}) fixed while changing the between-timestep behavior. The last paper explicitly finds that policy behavior between physics/control substeps matters, and that recomputing low-level control at the physics frequency generally improves performance for most action spaces, especially position control due to stiffness/damping effects. 

In your case, since you cannot directly recompute Unitree’s internal torque, the deployable analogue is:

```text
do not send raw step changes in q
send a smoother q_cmd trajectory at 1 kHz
```

---

## Priority 2: Do not use “torque-rate limiting after PD” as the main plan

My earlier torque-rate-limiter suggestion needs to be qualified for your stack.

If you owned the 1 kHz torque controller, you could do:

[
\tau_{\text{cmd}}[n]
====================

\tau_{\text{cmd}}[n-1]
+
\operatorname{clip}
\left(
\tau_{\text{raw}}[n]-\tau_{\text{cmd}}[n-1],
-\dot{\tau}*{\max}\Delta t,
\dot{\tau}*{\max}\Delta t
\right).
]

But your deployment code does not appear to own (\tau_{\text{raw}}). The Unitree interface exposes a feedforward `tau` field, but the position-control torque from (q,dq,kp,kd) is generated internally. The SDK examples show `tau` being written alongside `q`, `dq`, `kp`, and `kd`, but that does not mean you can clamp the firmware’s internal PD torque after it is computed. ([GitHub][2])

So the practical replacement is:

```text
limit q_cmd velocity
limit q_cmd acceleration
optionally limit q_cmd jerk
optionally reduce kp or modify kd
optionally limit feedforward tau if used
```

That attacks torque smoothness indirectly through the PD input.

Given your motor-side law is effectively:

[
\tau \approx K_p(q_{\text{cmd}}-q) + K_d(dq_{\text{cmd}}-\dot q) + \tau_{\text{ff}},
]

a discontinuity in (q_{\text{cmd}}) produces an immediate proportional torque jump:

[
\Delta \tau \approx K_p \Delta q_{\text{cmd}}.
]

So target conditioning is not cosmetic. It directly bounds proportional torque jumps.

---

# Concrete deployment change I would make first

## Add a 1 kHz joint target rate limiter

At each 1 kHz FSM tick:

[
q_{\text{cmd}}[n]
=================

q_{\text{cmd}}[n-1]
+
\operatorname{clip}
\left(
q_{\text{policy}}[k]-q_{\text{cmd}}[n-1],
-\dot q_{\max}\Delta t,
\dot q_{\max}\Delta t
\right)
]

where:

[
\Delta t = 0.001.
]

Then publish:

```cpp
lowcmd->msg_.motor_cmd()[motor_id].q() = q_cmd[i];
```

This is safer than a blind low-pass filter because you can reason about the maximum target velocity.

Then optionally compute a consistent desired velocity:

[
dq_{\text{cmd}}[n] = \frac{q_{\text{cmd}}[n] - q_{\text{cmd}}[n-1]}{\Delta t}
]

but I would test both:

```text
A: dq = 0
B: dq = generated dq_cmd
```

because Unitree’s interpretation of `dq` with nonzero `kp/kd` can matter. The public examples commonly set `dq = 0` while commanding positions. ([GitHub][2])

---

# Second deployment change: acceleration or jerk limiting

If velocity limiting alone still gives audible/visible jitter, use a second-order limiter:

[
\dot q_{\text{cmd}}[n]
======================

\dot q_{\text{cmd}}[n-1]
+
\operatorname{clip}
\left(
\dot q_{\text{desired}}-\dot q_{\text{cmd}}[n-1],
-\ddot q_{\max}\Delta t,
\ddot q_{\max}\Delta t
\right)
]

[
q_{\text{cmd}}[n] = q_{\text{cmd}}[n-1] + \dot q_{\text{cmd}}[n]\Delta t.
]

This gives you smooth (q_{\text{cmd}}) and bounded (\dot q_{\text{cmd}}). Since torque depends on position error and velocity error, this is usually a better intervention than lowering the ONNX call rate.

---

# Third deployment change: gain sweep, but with the right expectation

I would not blindly reduce gains. I would run a small structured sweep:

```text
kp scale: 1.0, 0.75, 0.5
kd scale: 1.0, 0.75, 0.5, 1.25
target rate limit: off, loose, medium, tight
```

Why: high (K_p) converts target jitter into torque jitter, while high (K_d) can suppress motion but amplify velocity-estimation noise or motor-side derivative effects. *Tune to Learn* also warns against evaluating gains in isolation: stiff/overdamped gains gave lower system-ID error but worse closed-loop sim-to-real transfer, with high-frequency oscillation as the main failure mode. ([arXiv][3])

That paper’s deployment setup is structurally relevant: it used a 1 kHz inner impedance loop and a 50 Hz learned outer loop, with torques clamped and torque rates limited before commanding the robot. ([arXiv][3]) But because you do not own the internal Unitree torque computation, your corresponding deploy-side knob is target/gain/feedforward conditioning, not post-PD torque-rate limiting.

---

# What to do with 25 Hz

## Case 1: Existing 50 Hz ONNX at 25 Hz

I would use this only as a **diagnostic ablation**, not the final fix.

Change:

```yaml
step_dt: 0.04
```

and leave ONNX/action scale/joint order/gains unchanged.

This tests the hypothesis:

```text
The policy is reacting too quickly to transient hardware states,
so fewer policy updates reduce closed-loop oscillation.
```

This is plausible. *Tune to Learn* reports that lowering policy frequency reduced oscillation because joints had more time to settle before the next action. Their policy-frequency ablation varied how long position commands were zero-order-held; they retrained policies for 10, 20, 50, and 100 Hz, while keeping the real-world controller rate fixed. ([arXiv][3])

But deploying a 50 Hz-trained policy at 25 Hz is a distribution shift. Your own stack makes that especially true if observations include `last_action`, phase, command resampling counters, or implicit contact timing learned at 20 ms intervals.

So I would run:

```text
Test A: 50 Hz policy, 50 Hz deploy, no target conditioning
Test B: 50 Hz policy, 50 Hz deploy, target conditioning
Test C: 50 Hz policy, 25 Hz deploy, no target conditioning
Test D: 50 Hz policy, 25 Hz deploy, target conditioning
```

If C improves smoothness but hurts tracking, that is evidence for retraining at 25 Hz rather than permanently throttling the 50 Hz policy.

## Case 2: Retrain at 25 Hz

Your proposed retraining change is correct:

```python
sim_dt = 0.005
decimation = 8
```

This gives:

```text
policy:       25 Hz
sim PD:      200 Hz
physics:     200 Hz
```

This is the right way to preserve the “slow policy, faster low-level tracking” structure.

Also correct:

```yaml
step_dt: 0.04
```

for deployment.

And yes, update real-time-count parameters:

```text
command_resample_interval: 250 -> 125  # preserve 5 s
max_episode_steps: 1000 -> 500         # preserve 20 s
```

unless you intentionally want longer real-time episodes.

The papers’ convention supports this: *Tune to Learn* retrained policies for new policy frequencies, and *Action Space Design* trained/evaluated policies across 25–200 Hz with fixed physics frequency and frame skips. ([arXiv][3])

---

# Updated experimental plan

## Phase 0: Instrumentation

Log at the highest rate available:

```text
policy time
q_policy_raw
q_cmd_published
dq_cmd_published
kp, kd, tau_ff
q_measured
dq_measured
tau_est if available
foot/contact estimate
base angular velocity
policy inference interval jitter
LowCmd publish interval jitter
```

Compute:

```text
||q_policy[k] - q_policy[k-1]||
||q_cmd[n] - q_cmd[n-1]||
estimated proportional torque jump: kp * (q_cmd[n] - q_cmd[n-1])
tracking error: q_cmd - q_measured
velocity noise spectrum
base angular velocity spectrum
```

This tells you whether the jitter is from raw policy action jumps, command publishing irregularity, derivative/noisy velocity, contact transitions, or firmware tracking.

---

## Phase 1: Keep 50 Hz policy, add 1 kHz target conditioning

Recommended first variants:

```text
V0: raw q_policy held at 1 kHz
V1: q velocity limit only
V2: q velocity + acceleration limit
V3: first-order low-pass on q
V4: q limiter + modest kp reduction
```

I would prioritize V1 and V2 over V3.

The first-order low-pass is:

[
q_{\text{cmd}}[n]
=================

q_{\text{cmd}}[n-1]
+
\alpha(q_{\text{policy}}[k]-q_{\text{cmd}}[n-1])
]

[
\alpha = 1-\exp(-\Delta t/\tau_f).
]

But it is less interpretable than a rate/acceleration limiter because the induced delay depends on frequency content.

---

## Phase 2: 50 Hz policy deployed at 25 Hz as ablation

Change only:

```yaml
step_dt: 0.04
```

Use this to answer:

```text
Does slower outer-loop reaction reduce oscillation?
```

Do not call this a valid 25 Hz policy unless it still passes tracking, stability, and disturbance tests.

---

## Phase 3: Retrain 25 Hz policy

If Phase 2 reduces jitter substantially, retrain with:

```python
sim_dt = 0.005
decimation = 8
```

Then deploy with:

```yaml
step_dt: 0.04
```

Also update `test_onnx_policy.py` to instantiate the environment with `decimation = 8`, otherwise ONNX testing is not testing the same timing MDP.

---

## Phase 4: Train with deploy-side target conditioning included

For the most faithful setup, mirror the exact target conditioner in training:

```python
policy action at 25 or 50 Hz
q_policy = action_to_target(action)

for sim substep:
    q_cmd = target_conditioner(q_policy, q_cmd_prev)
    tau = kp * (q_cmd - q_current) + kd * (0 - dq_current)
    mujoco.mj_step(...)
```

This matters because a policy trained with instantaneous target steps may learn to exploit those sharp transitions. If you filter only at deployment, you add phase lag and reduce effective action authority without giving the policy a chance to adapt.

---

# Final recommendation

Your first fix should **not** be retraining at 25 Hz. It should be:

```text
Keep policy at 50 Hz.
Keep LowCmd publishing at 1 kHz.
Insert q_cmd target conditioning in State_RLBase.cpp before writing LowCmd.q.
Log q_policy, q_cmd, q, dq, and tau_est/proxy.
Then test 25 Hz deployment only as a diagnostic.
Retrain at 25 Hz only if the 25 Hz diagnostic materially improves smoothness.
```

The reason is specific to your stack: you do not directly command post-PD torque, so the cleanest deploy-side smoothness lever is the trajectory you feed to Unitree’s internal motor PD, not the ONNX model itself.

[1]: https://deepwiki.com/unitreerobotics/unitree_sdk2/7-low-level-motor-control "Low-Level Motor Control | unitreerobotics/unitree_sdk2 | DeepWiki"
[2]: https://github.com/unitreerobotics/unitree_sdk2/blob/main/example/go2/go2_stand_example.cpp "unitree_sdk2/example/go2/go2_stand_example.cpp at main · unitreerobotics/unitree_sdk2 · GitHub"
[3]: https://arxiv.org/html/2604.02523v1 "Tune to Learn: How Controller Gains Shape Robot Policy Learning"
