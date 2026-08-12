# Real-Hardware Deployments
This document is just reference for running all three controllers (MPC-Injection, Reward-shaping, and AMP-MPC) for the paper's testing. For more detailed instructions for running the MPC-Injection and Reward shaping policies please refer below:

For Go2 controller setup and operating instructions, see
[`deploy/README.md`](../deploy/README.md).

## MPC-Injection

From `~/MPC-RL`, activate the export environment and export the policy:

```bash
conda activate mpc-rl
python deploy/export_onnx_go2.py --model_zip logs/quadruped_domain_rand_mpc_dr_sysid_dyn20_mjlab_10k_LPF/percentage_ablations/quadruped-velocity_tracking-SAC-MPC-20260520-180440-percentage-30pct-seed400-env256-drsysid_dyn20_mjlab/checkpoints/model_900000_steps.zip --vecnorm_pkl logs/quadruped_domain_rand_mpc_dr_sysid_dyn20_mjlab_10k_LPF/percentage_ablations/quadruped-velocity_tracking-SAC-MPC-20260520-180440-percentage-30pct-seed400-env256-drsysid_dyn20_mjlab/checkpoints/model_vecnormalize_900000_steps.pkl --output_dir deploy/robots/go2/config/policy/velocity/v0/exported/
```

Launch from `~/MPC-RL/deploy/robots/go2/build`:

```bash
./go2_ctrl --network=enp5s0 --torque_output ~/go2_traces/mpc_injection_tau_est.npz
```

## Reward-Shaping

From `~/MPC-RL`, activate the export environment and export the policy:

```bash
conda activate mpc-rl
python deploy/export_onnx_go2.py --model_zip logs/quadruped_td3_lpf/quadruped-velocity_tracking-TD3-20260514-051543-seed805-env256-drsysid_dyn20_mjlab/final_model.zip --vecnorm_pkl logs/quadruped_td3_lpf/quadruped-velocity_tracking-TD3-20260514-051543-seed805-env256-drsysid_dyn20_mjlab/vec_normalize.pkl --output_dir deploy/robots/go2/config/policy/velocity/v0/exported/
```

Run it with the same `go2_ctrl` command and default deployment setup as
MPC-Injection:

```bash
cd ~/MPC-RL/deploy/robots/go2/build
./go2_ctrl --network=enp5s0 --torque_output ~/go2_traces/reward_shaping_tau_est.npz
```

## AMP-MPC

### Launch

Selected actor-only TorchScript policy:

```text
/home/roy/AMP_mpc_results/Aug07_14-56-55_real_8814_pd2040_exact_seed200_3072env_cluster_50000iter/policy_model_50000.pt
```

Use this file, not the full `model_50000.pt` training checkpoint.

The selected policy used:

```text
Kp: [20, 20, 40] per leg (hip, thigh, calf)
Kd: [ 1,  1,  2] per leg (hip, thigh, calf)
vx: [0.0, 0.5] m/s; vy: 0; yaw rate: 0
```

The launcher accepts scalar or 12-element gains and maps training-order gains
to Unitree SDK joint order. Use the dedicated AMP-MPC config, which sets the
trained walk gains and restricts commands to forward velocity `[0.0, 0.5]`:

```bash
cd ~/AMP_docker_env/legged_rl_gym/deploy
conda activate amp-go2-sdk2
python sim2real_go2_sdk2_keyboard.py --network enp5s0 --config config/go2_amp_mpc.yaml --torque_output ~/go2_traces/amp_mpc_tau_est.npz
```

### Verified hardware transport

Real-robot communication runs directly on the host, not in Docker or
Apptainer. Use the existing `amp-go2-sdk2` Conda environment and Unitree SDK2
over CycloneDDS.

1. Connect the workstation directly to the Go2 over Ethernet, put the robot in
   a harness, and activate the saved Go2 network profile:

   ```bash
   nmcli connection up uuid b6073e88-7d75-4a3a-938c-6c7bf2166732
   ```

   Use the saved Go2 network profile on `enp5s0` with host address
   `192.168.123.99/24`.

2. Stop every other low-level controller. Never run `go2_ctrl` at the same
   time as the AMP SDK2 launcher.

3. Run the launch command above. Do not use `config/go2.yaml` for this policy.

### Operation

The SDK2 script releases Unitree's active high-level motion mode itself. It
starts in `FixStand` and follows the working MPC-RL stand trajectory: one
second to `[0, 1.36, -2.65]` per leg, then one second to
`[0, 0.9, -1.8]`. It holds the final pose with the MPC-RL stand gains. Set the
desired velocity command while the robot remains in `FixStand`, then press
`P` to initialize AMP policy control. The policy does not start automatically.

- `I` or `8`: increase forward command by `0.1 m/s`, up to `0.5 m/s`.
- `K` or `2`: decrease forward command.
- `P`: transition from `FixStand` to AMP policy control after the stand ramp.
- `Space` or `5`: set velocity to zero; this is **not** an emergency stop.
- `Q` or `Esc`: send damping commands and exit.

SDK2 transport and Go2 low-state reception have been verified. The explicit
`FixStand`-to-policy transition has not yet been exercised on hardware.

## Hardware torque traces

`--torque_output` is optional. It records Unitree SDK2 low-state
`motor_state[].tau_est` feedback, not the zero feed-forward torque command.
Use one `.npz` path for each FixStand-to-policy activation; a later activation
using the same path overwrites it. Samples begin only when MPC-Injection or
Reward-Shaping enters Velocity, or AMP-MPC accepts `P` from ready FixStand.
FixStand, Passive, and damping are excluded. Leaving policy for Passive,
safety exit, or `Ctrl+C` finalizes a non-empty archive; no archive is written
when policy never begins.

The archive keeps `tau_applied` for compatibility, but its contents are
hardware estimated output torque (`tau_est`) in `FL, FR, RL, RR` training
order. To compare a recorded hardware run with a simulated trajectory:

```bash
cd ~/MPC-RL
python body_trajs/plot_torque_cdf_quadruped.py \
  --trajectory /path/to/simulated_trajectory.npz \
  --trajectory ~/go2_traces/mpc_injection_tau_est.npz \
  --no-show
```
