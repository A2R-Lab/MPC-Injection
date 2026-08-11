# Go2 barrel roll with MPC-injected SAC: G9 results

## Experimental setup

We trained two Soft Actor-Critic policies for a Unitree Go2 barrel roll in MuJoCo. Training used four parallel environments, 500,000 environment steps per training seed, and a replay buffer maintained at 25% MPC transitions (`SAC-MPC`). The policies used a 45-dimensional actor observation and a 4-dimensional privileged critic observation. Domain randomization was disabled and the identified Go2 joint-dynamics model was enabled. The control and simulation rates were 50 Hz and 200 Hz, respectively; each episode lasted 2.50 s (125 control or 500 physics steps). Training seeds 1 and 2 were evaluated independently.

The initial state randomized all four hip joints with one symmetric spread sampled uniformly from \([0,0.10]\) rad. The desired positive roll started at 0.20 s and reached \(2\pi\) at 0.80 s using a cubic smooth-step. If

\[
u(t)=\operatorname{clip}\!\left(\frac{t-0.20}{0.60},0,1\right),
\]

then the desired roll and roll rate were

\[
\phi_d(t)=2\pi u(t)^2(3-2u(t)), \qquad
\dot\phi_d(t)=
\begin{cases}
\dfrac{12\pi}{0.60}u(t)(1-u(t)), & 0<u(t)<1,\\
0, & \text{otherwise}.
\end{cases}
\]

## Reward and termination

The implemented per-control-step reward was:

```python
desired_roll = desired_roll_at_time(step_count * control_dt)
roll_error = desired_roll - roll_progress
roll_tracking = np.exp(-(roll_error / 0.50) ** 2)

desired_angular_velocity = np.array([
    desired_roll_rate_at_time(step_count * control_dt), 0.0, 0.0
])
rate_error = np.linalg.norm(base_angular_velocity - desired_angular_velocity)
rate_tracking = 0.25 * np.exp(-(rate_error / 1.0) ** 2)

action_change = -0.02 * np.linalg.norm(action - previous_action) ** 2
terminal_outcome = 0.0
if terminated:
    terminal_outcome = 50.0 if failure_reason is None else -50.0

reward = roll_tracking + rate_tracking + action_change + terminal_outcome
```

Equivalently, for action \(a_t\), unwrapped roll progress \(\phi_t\), and base angular velocity \(\boldsymbol\omega_t\),

\[
r_t=
\exp\!\left[-\left(\frac{\phi_t-\phi_d(t)}{0.50}\right)^2\right]
+0.25\exp\!\left[-\left(\frac{\lVert\boldsymbol\omega_t-[\dot\phi_d(t),0,0]^\top\rVert_2}{1.0}\right)^2\right]
-0.02\lVert a_t-a_{t-1}\rVert_2^2+b_t,
\]

where \(b_t=0\) before termination, \(+50\) for terminal success, and \(-50\) for terminal failure. No translational-speed, joint-speed, torque, contact, or support term was used.

Episodes did not terminate early for falls or contacts. Except for a non-finite physics state, every episode terminated at step 125. Terminal success required all three conditions:

\[
1.75\pi \leq \phi_T \leq 2.50\pi, \qquad
h_T \geq 0.16\ \mathrm{m}, \qquad
\theta_{\mathrm{up},T}\leq \pi/3.
\]

Here \(h_T\) is base height and \(\theta_{\mathrm{up},T}\) is the angle between body-up and world-up. Failure precedence was non-finite state, rotation outside the accepted band, low base height, then excessive tilt. Foot contact, contact streaks, velocity, and non-foot ground contact were diagnostic only.

## MPC trajectories and replay injection

The schema-v3 demonstration set contains 1,000 successful trajectories and 125,000 control transitions. MPX replanned at 50 Hz control boundaries during the 1.40 s maneuver; a landing-stability latch could stop replanning after 0.86 s, and replanning never continued at or after 1.40 s. The controller then repeated the terminal-padded final-stance plan, with the same PD feedback, through the 2.50 s episode. Controller torques were applied directly at 200 Hz; replay actions were clipped inverse-PD residual labels. The online RL environment retained a 5 Hz target low-pass filter and action scale 2.0, so the dataset deliberately records `saved_action_reproduces_lpf_transition=False`.

Four workers required 1,195 attempts to accept 1,000 terminal successes (83.68%). Rejections were 114 rotation-out-of-band, 61 non-finite solver output, 15 low-height, and 5 excessive-tilt cases. Accepted trajectories had median terminal roll progress 6.182 rad, height 0.259 m, and body-up tilt 0.187 rad. Their median cumulative roll- and rate-tracking rewards were 106.55 and 23.06; the median cumulative action-change term was -0.690. Its magnitude was 0.535% of the median positive dense reward. The dataset occupied 4.152 GB and was checksum-validated before training.

## Policy selection and results

Checkpoint validation used the same 100 fixed seeds (2,000,000--2,000,099) every 10,000 training steps, including step zero. Selection used success rate only and retained the earliest checkpoint attaining each run's maximum. Both runs first reached 100/100 validation success, at 220,000 steps for training seed 1 and 230,000 steps for training seed 2. Later performance was non-monotonic; both reached 0/100 at step 440,000, so the final 500,000-step models were not substituted for the selected checkpoints.

After selection was locked, each selected policy was evaluated exactly once on the untouched seeds 3,000,000--3,000,099.

| Training seed | Selected step | Validation success | Untouched final success | Final failures |
| ---: | ---: | ---: | ---: | --- |
| 1 | 220,000 | 100/100 (100%) | **100/100 (100%)** | None |
| 2 | 230,000 | 100/100 (100%) | **98/100 (98%)** | One low-height; one rotation-out-of-band |

Seed 1 was designated the official policy by final-test success rate. Ten explicit rerenders per training seed on final-test seeds 3,000,000--3,000,009 were all successful. These videos are confirmations of recorded outcomes, not additional independent final-test trials.

## Scope and limitations

These are simulation-only results; no real-robot validation was performed. The terminal classifier permits residual motion: the official 100-episode final set included terminal base angular speeds up to 7.496 rad/s and joint-velocity norms up to 24.643. The 20 reviewed videos show a lateral roll and upright return, but the success definition does not impose terminal velocity or contact constraints. The inverse-PD replay labels also do not exactly reconstruct direct-torque MPC transitions through the online 5 Hz action filter. A concurrently running 0%-MPC baseline is not included here because it was incomplete when this report was written.
