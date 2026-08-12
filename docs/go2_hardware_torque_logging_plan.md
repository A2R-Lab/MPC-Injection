# Go2 hardware torque logging plan

## Objective

Collect one real-hardware torque trace per manually driven policy activation,
then pass the traces directly to
`body_trajs/plot_torque_cdf_quadruped.py` to produce CDFs comparable in form
to the existing simulation plots.

The three deployment paths are:

| Controller | Policy format | Deployment entry point |
|---|---|---|
| MPC-Injection | ONNX | `deploy/robots/go2/build/go2_ctrl` |
| Reward-Shaping | ONNX | `deploy/robots/go2/build/go2_ctrl` |
| AMP-MPC | TorchScript | `AMP_docker_env/legged_rl_gym/deploy/sim2real_go2_sdk2_keyboard.py` |

MPC-Injection and Reward-Shaping share one C++ implementation. AMP-MPC needs
the equivalent Python implementation.

## Confirmed design decisions

- Logging is opt-in with an optional explicit
  `--torque_output <path>.npz` argument in both launchers. Existing launch
  commands retain their behavior when it is omitted.
- A supplied output path may be overwritten. Do not add output-path collision
  protection or run-management logic.
- A recording is one Policy/Velocity activation and produces one CDF curve.
  Do not implicitly pool multiple runs.
- Start collecting only when the controller leaves `FixStand` and policy
  control begins. Exclude FixStand, Passive, and damping samples.
- Capture every received `rt/lowstate` message, with no collection-time
  downsampling. Include the Go2 low-state tick and a host monotonic timestamp.
- Capture the velocity command at each policy-inference boundary as ancillary
  data. It does not affect the CDF calculation.
- Finalize the output on a normal exit from Policy/Velocity, policy safety
  abort, or terminal `Ctrl+C`. Crash-recovery checkpoints are out of scope.
- Use Unitree SDK2 `motor_state[].tau_est` / `tau_est()` (estimated output
  torque in Nm), not the command field: all current position-PD deployments
  set command feed-forward `tau` to zero.
- Save SDK2 samples in training order: `FL, FR, RL, RR`, with hip, thigh,
  calf within each leg. Reuse each controller's existing SDK-to-training map.
- Use an `.npz` archive with the existing required `tau_applied` key and
  `(12, samples)` orientation. The existing CDF script must continue to read
  all historical simulation files unchanged.
- Real curves must explicitly identify the source as hardware estimated output
  torque (`tau_est`); do not claim it is identical to simulation's applied
  controller torque.

## Archive contract

Both writers should emit the following arrays:

| Key | Type and shape | Meaning |
|---|---|---|
| `tau_applied` | `float32`, `(12, N)` | Hardware `tau_est` in Nm, training order. This name is retained for CDF compatibility. |
| `lowstate_tick` | `uint32`, `(N,)` | Tick from each captured Go2 low-state message. |
| `host_monotonic_ns` | `int64`, `(N,)` | Host monotonic time when each feedback message was captured. |
| `commanded_velocity` | `float32`, `(3, M)` | Policy command `[vx, vy, yaw_rate]` consumed at each policy inference. |
| `command_host_monotonic_ns` | `int64`, `(M,)` | Timestamp for each policy command sample. |
| `is_hardware_tau_est` | `uint8`, `(1,)`, value `1` | Allows plotting code to identify the torque source without relying on paths. |

If policy control never begins, do not write an empty archive. Report that no
Policy/Velocity samples were collected.

## C++ implementation: MPC-Injection and Reward-Shaping

1. Extend `deploy/include/param.h` and `deploy/robots/go2/main.cpp` to accept
   the optional `--torque_output` path and announce whether recording is
   enabled before FixStand begins.
2. Add a small application-owned recorder module under the existing Go2
   controller source/include tree. It must create an independent SDK2
   `rt/lowstate` subscriber, append a copy of each message's twelve `tau_est`
   values only while an atomic recording flag is true, and protect its buffer
   against concurrent callback and finalization access.
3. Reorder callback samples with the active policy's existing
   `joint_ids_map`, rather than hard-coding a second mapping.
4. Start the recorder in `State_RLBase::enter()` and stop/finalize it in
   `State_RLBase::exit()`. No command, gain, filter, target, or state-machine
   behavior may change.
5. Preserve the exact command that produces each policy observation. The
   current `velocity_commands` observation helper should expose or retain that
   three-vector during `ManagerBasedRLEnv::step()`, and `State_RLBase` should
   append it once per policy step. Do not reconstruct it later from a changed
   joystick state.
6. Add orderly `Ctrl+C` handling: the signal handler only requests shutdown;
   normal program context stops the FSM/policy worker as required, disables
   collection, and writes the archive. Do not allocate, lock, or write files
   from the signal handler.
7. Integrate the existing vendored `deploy/thirdparty/cnpy` source into the
   Go2 CMake target (including zlib) so C++ writes the same `.npz` contract
   without adding a new external dependency.

## Python implementation: AMP-MPC

1. Add optional `--torque_output` parsing to
   `deploy/sim2real_go2_sdk2_keyboard.py` and pass it into the controller.
2. Extend the existing low-state callback to copy `tau_est`, low-state tick,
   and `time.monotonic_ns()` into a recorder only when Policy is active.
3. Start recording immediately after a valid `P` transition from ready
   FixStand. Record the local `commands` vector exactly where AMP policy
   inference consumes it.
4. In the existing `finally` shutdown flow, disable collection, write a
   compressed `.npz` if samples exist, print the destination and summary, then
   retain the existing damping behavior. `KeyboardInterrupt` must follow this
   same path.

## Plotting and documentation

1. Update `body_trajs/plot_torque_cdf_quadruped.py` only to detect
   `is_hardware_tau_est` and label a real trace as hardware estimated output
   torque. Keep the existing `tau_applied` calculation, simulation labels,
   flags, and all historical `.npz` behavior unchanged.
2. Update `docs/DEPLOY_REAL_HARDWARE.md` after implementation with
   `--torque_output` examples for all three controllers, the one-activation/
   one-file convention, the FixStand-to-policy recording boundary, `Ctrl+C`
   finalization, and a CDF overlay command.

## Validation

1. Build the Go2 C++ controller with its CNPY integration.
2. Byte-compile and import the AMP launcher in `amp-go2-sdk2`.
3. Add focused non-hardware tests or mocks for both recorders covering:
   no pre-policy samples, correct SDK-to-training ordering, torque values,
   tick/timestamp alignment, command capture, normal finalization, and
   `Ctrl+C` shutdown handling.
4. Load generated archives with NumPy and run the existing CDF script with
   `--no-show`. Verify both a simulated archive and a hardware-marked archive
   remain readable and get the correct source labels.
5. Hardware validation in a harness: for each policy set the desired command
   while in FixStand, activate policy, manually drive, leave policy or press
   `Ctrl+C`, verify the saved archive has nonzero Policy samples only, then
   generate one CDF curve per run.

## Scope and risk

This is recording-only work. Do not alter policy startup, PD gains, action
filtering, target clipping, joint limits, or policy binaries. Preserve all
existing user working-tree changes. The central measurement limitation is that
hardware `tau_est` is Unitree feedback-estimated output torque, whereas
simulation records the simulator's applied controller torque; the result is a
useful Nm-distribution comparison but not an assertion that the two signals
are identical.
