# Deployment Code for the Go2

This code is based on the [Unitree MjLab deployment code](https://github.com/unitreerobotics/unitree_rl_mjlab).

---

## How the deployment pipeline works
For reference!

### Training -> real robot: the four-step chain

```
Python training  ->  ONNX export  ->  C++ compile  ->  real robot
 (PyTorch/SB3)       (.onnx file)    (C++ binary)    (50 Hz loop)
```

### Step 1 -- What gets trained

We train an SAC or TD3 policy in `mpc_rl/` using Stable-Baselines3 (SB3).
The algorithm is recorded in `config.json` inside the run directory and is
also encoded in the directory name (e.g. `...-SAC-...` or `...-TD3-...`).

Both algorithms share the same observation and action spaces, but their
**actor network structures differ** (this matters for ONNX export):

#### SAC actor -- two separate sub-modules

- **`model.policy.actor.latent_pi`** -- the MLP trunk:
  ```
  Linear(45 -> 256) -> ReLU -> Linear(256 -> 256) -> ReLU
  ```
- **`model.policy.actor.mu`** -- the mean output head (no activation):
  ```
  Linear(256 -> 12)
  ```
  The full SAC inference is:
  `normalise -> latent_pi -> mu -> tanh()` (tanh applied externally)

#### TD3 actor -- one monolithic module

- **`model.policy.actor.mu`** -- the *entire* network including tanh:
  ```
  Linear(45 -> 400) -> ReLU -> Linear(400 -> 300) -> ReLU -> Linear(300 -> 12) -> Tanh()
  ```
  There is **no** `latent_pi` attribute. The full TD3 inference is:
  `normalise -> mu` (tanh is the last layer inside `mu`)

Both produce actions in `[-1, 1]`. The C++ action manager
(`JointPositionAction` in `deploy/include/isaaclab/envs/mdp/actions/joint_actions.h`)
is the component that converts raw neural network outputs into physical joint
position targets. It reads `scale` and `offset` arrays from `deploy.yaml` and
applies them element-wise to each action dimension. For the Go2, this becomes:
```
joint_target_raw[i] = action[i] * 0.5 + default_joint_pos[i]
joint_target[i] = action_lpf(joint_target_raw[i])
```
identically for both algorithms. The low-pass-filtered targets are then written to the
Unitree SDK `LowCmd` motor position fields at 1 kHz by `State_RLBase::run()`.

### Step 2 -- Observation normalisation and why it must be "baked in"

During training, SB3's **`VecNormalize`** wrapper sits around the environment
and keeps a running mean and variance of every observation.  Before passing
observations to the neural network it applies:
```
x_normalised[i] = clip( (x_raw[i] - mean[i]) / std[i], -10, +10 )
```

The neural network therefore **only ever sees normalised values**.  If you
feed it raw IMU/encoder outputs without normalisation the actions will be
garbage.

The C++ deployment binary has no Python and no `VecNormalize`.  The solution
used in `deploy/export_onnx_go2.py` is to **bake the normalisation constants
into the ONNX graph** as frozen constant tensors.  The exported ONNX model
does the normalisation step first, then feeds the result into `latent_pi` and
`mu`.  From the C++ side this is invisible: it simply passes raw sensor values
in and gets joint targets out.

### Step 3 -- The ONNX file

ONNX (Open Neural Network Exchange) is a universal format for inference
engines.  The C++ deployment binary links against **ONNXRuntime**, a library
that can load any `.onnx` file and run inference on CPU or GPU without needing
Python.

The exported `policy.onnx` encodes the following computation. For **SAC**:
```
obs (1x45, raw sensors)
  -> normalise: clip((obs - mean) / std, -10, 10)
  -> latent_pi: Linear(45->256) -> ReLU -> Linear(256->256) -> ReLU
  -> mu:        Linear(256->12)
  -> tanh
  -> actions (1x12, values in [-1, 1])
```
For **TD3**:
```
obs (1x45, raw sensors)
  -> normalise: clip((obs - mean) / std, -10, 10)
  -> mu:        Linear(45->400) -> ReLU -> Linear(400->300) -> ReLU -> Linear(300->12) -> Tanh
  -> actions (1x12, values in [-1, 1])
```

The ONNX graph is identical from the C++ side regardless of the source
algorithm: one input named `"obs"`, one output named `"actions"`.  The
`OrtRunner` class reads these names from the ONNX metadata dynamically.

### Step 4 -- The C++ deploy binary

`deploy/robots/go2/` contains a CMake C++ program that:

1. Connects to the Go2 over DDS (`unitree_sdk2`).
2. Reads `config/config.yaml` to set up a finite-state machine (FSM):
   - **Passive** -- all joints loose (safe starting state)
   - **FixStand** -- interpolates to a standing pose
   - **Velocity** -- runs the RL policy at 50 Hz
3. When in **Velocity** mode it:
   a. Reads `LowState` from the SDK (IMU + joint encoders)
   b. Assembles the 45-dim observation vector
   c. Calls the ONNX policy -> 12 raw actions
   d. Scales: `joint_target_raw = action * 0.5 + default_joint_pos`
   e. Low-pass filters the joint target before the motor-side PD controller
   f. Writes position targets to `LowCmd` which the SDK sends to the motors
4. PD gains (stiffness / damping) are set from `config/policy/velocity/v0/params/deploy.yaml`.

### Observation vector layout (45 dimensions)

| Slice | Name | Source | Dim |
|-------|------|--------|-----|
| `[0:3]`   | `base_ang_vel`    | IMU gyroscope (body frame) | 3 |
| `[3:6]`   | `projected_gravity` | IMU orientation -> R^T*[0,0,-1] | 3 |
| `[6:9]`   | `velocity_commands` | Velocity command source (controller vx/vy/wz or keyboard vx-only) | 3 |
| `[9:21]`  | `joint_pos_rel`   | Encoder pos - default pos | 12 |
| `[21:33]` | `joint_vel_rel`   | Encoder velocities | 12 |
| `[33:45]` | `last_action`     | Previous policy output | 12 |

### Joint Ordering

The exported MPC-RL policy uses the same joint order as the MuJoCo training
environment: **FL -> FR -> RL -> RR** (front-left, front-right, rear-left,
rear-right), each with hip -> thigh -> calf.

Unitree's Go2 SDK motor order is **FR -> FL -> RR -> RL**. The mapping in
`deploy.yaml` converts SDK motor order into the policy/training order:

```yaml
joint_ids_map: [3,4,5, 0,1,2, 9,10,11, 6,7,8]
# policy idx:   FL      FR      RL       RR
# SDK2 motor:   FL      FR      RL       RR  (re-indexed from FR/FL/RR/RL)
```

---

## Quick-start: full workflow

### Prerequisites

```bash
# Install cyclonedds and unitree_sdk2 per their respective READMEs
# then activate the conda environment
conda activate mpc-rl
```

---

### Step A -- Export the policy to ONNX

Run from the **MPC-RL root directory**.
The algorithm (SAC or TD3) is **auto-detected** from the directory name.
Use `--algo SAC` or `--algo TD3` to override.

**SAC** -- using the final checkpoint of the 5M-step run:
```bash
python deploy/export_onnx_go2.py \
    --model_zip  logs/quadruped-velocity_tracking-SAC-20260305-170808/final_model.zip \
    --vecnorm_pkl logs/quadruped-velocity_tracking-SAC-20260305-170808/vec_normalize.pkl \
    --output_dir deploy/robots/go2/config/policy/velocity/v0/exported/
```

**TD3** -- same invocation, just point to a TD3 run directory:
```bash
python deploy/export_onnx_go2.py \
    --model_zip  logs/quadruped-velocity_tracking-TD3-<timestamp>/final_model.zip \
    --vecnorm_pkl logs/quadruped-velocity_tracking-TD3-<timestamp>/vec_normalize.pkl \
    --output_dir deploy/robots/go2/config/policy/velocity/v0/exported/
```

For a specific training checkpoint instead of the final model:
```bash
python deploy/export_onnx_go2.py \
    --model_zip  logs/quadruped-velocity_tracking-SAC-20260305-170808/checkpoints/model_4988928_steps.zip \
    --vecnorm_pkl logs/quadruped-velocity_tracking-SAC-20260305-170808/checkpoints/model_vecnormalize_4988928_steps.pkl \
    --output_dir deploy/robots/go2/config/policy/velocity/v0/exported/
```

The script prints a full sanity-check: it runs ONNXRuntime on a zero obs and a
random obs and compares against the original PyTorch model to confirm the
export is numerically correct.

---

### Testing Multiple Policies

For real-robot sweeps, export each candidate into its own deploy directory
instead of repeatedly overwriting `v0/exported/policy.onnx`. Do this with the
batch exporter; do not manually set `RUN` for every training directory.

Run from the **MPC-RL root directory**:

```bash
python deploy/batch_export_onnx_go2.py --dry_run
```

The dry run scans:

```text
logs/quadruped_domain_rand_mpc_dr_sysid_dyn20_mjlab_10k_LPF/SAC-MPC-sysid_dyn20_mjlab/
```

and prints every deployable candidate it finds. By default, it looks for both:

```text
final_model.zip + vec_normalize.pkl
checkpoints/model_900000_steps.zip + checkpoints/model_vecnormalize_900000_steps.pkl
```

Then export every available candidate:

```bash
python deploy/batch_export_onnx_go2.py --network=enp130s0
```

Each output policy directory has the layout expected by `go2_ctrl`:

```text
deploy/robots/go2/config/policy/velocity/policies/<policy_name>/
  params/deploy.yaml
  exported/policy.onnx
```

The script skips incomplete runs, which is expected while training jobs are
still running. It also skips policies that are already exported and current, so
rerun the same command later to pick up newly finished final models:

```bash
python deploy/batch_export_onnx_go2.py --network=enp130s0
```

Useful options:

```bash
# Re-export even if policy.onnx already exists
python deploy/batch_export_onnx_go2.py --force --network=enp130s0

# Export another checkpoint in addition to 900k
python deploy/batch_export_onnx_go2.py --checkpoint_step=900000 --checkpoint_step=1200000

# Export only the 900k checkpoint candidates
python deploy/batch_export_onnx_go2.py --skip_final
```

After exporting, the script writes:

```text
deploy/robots/go2/config/policy/velocity/policies/manifest.tsv
deploy/robots/go2/config/policy/velocity/policies/launch_commands.txt
```

Use `manifest.tsv` to track which log directory produced each policy. Use
`launch_commands.txt` as the real-robot test queue.

Before running a candidate on the robot, test that exact ONNX file in
simulation:

```bash
python deploy/test_onnx_policy.py \
    --onnx deploy/robots/go2/config/policy/velocity/policies/<policy_name>/exported/policy.onnx
```

When launching the robot controller, select a specific policy with
`--policy_dir`. Relative paths are resolved from `deploy/robots/go2`, the Go2
controller project directory:

```bash
cd deploy/robots/go2/build
./go2_ctrl \
    --network=enp130s0 \
    --policy_dir=config/policy/velocity/policies/<policy_name>
```

If `--policy_dir` is omitted, the controller keeps using the `policy_dir` from
`deploy/robots/go2/config/config.yaml`, so the existing workflow still works.

---

### Step B -- Test the ONNX policy in simulation

Before touching the real robot, verify the ONNX policy drives the simulated
Go2 correctly using the same MuJoCo environment it was trained in:

`test_onnx_policy.py` uses the same Go2 simulator path as training/evaluation,
including the runtime-applied sysID joint dynamics from `deploy/sys_id/report.html`.
Domain randomization is disabled by default so this replay matches deterministic
policy evaluation semantics.

```bash
python deploy/test_onnx_policy.py \
    --onnx deploy/robots/go2/config/policy/velocity/v0/exported/policy.onnx
```

For a side-by-side numerical comparison against the original SB3 model (prints
max action difference every 50 steps -- should be < 1e-5):

```bash
python deploy/test_onnx_policy.py \
    --onnx    deploy/robots/go2/config/policy/velocity/v0/exported/policy.onnx \
    --compare logs/quadruped-velocity_tracking-SAC-20260305-170808/final_model.zip \
    --vecnorm logs/quadruped-velocity_tracking-SAC-20260305-170808/vec_normalize.pkl
```

Controls (press keys in the MuJoCo viewer window):

| Key | Command |
|-----|---------|
| Up / Down | forward / backward (vx) |
| Left / Right | strafe left / right (vy) |
| `[` / `]` | turn left / right (wz) |
| `R` | reset all to zero |
| `Esc` | quit |

---

### Step C -- Compile the C++ deployment binary

```bash
cd deploy/robots/go2
mkdir -p build && cd build
cmake .. && make
```

The compiled binary `go2_ctrl` will be at `deploy/robots/go2/build/go2_ctrl`.

---

### Step D -- Deploy to the real Go2

1. **Connect** your PC to the robot via Ethernet.
   - Your Address: `192.168.123.99`
   - Netmask: `255.255.255.0`
   Use `ifconfig` to find the Ethernet interface name (e.g. `enp5s0`).

2. **Power on** the robot in a safe state (use a harness).

3. **Do not run any other low-level controller.** `go2_ctrl` now uses SDK2's
   `MotionSwitcherClient` at startup to release Unitree's active high-level
   motion service (`sport_mode`, `ai_sport`, or `advanced_sport`) before it
   creates its own `rt/lowcmd` publisher. Keep `basic_service` enabled.

   Do not run any other low-level example or ROS low-level node at the same time
   as this binary, including `go2_stand_example`, `lowlevel_control.py`, or
   `go2_lowroscontrol`. The executable checks `rt/lowcmd` after releasing the
   Unitree service and exits if another publisher is still active.

4. **Enter the robot's safe handoff state and then debug mode**: Press
   `L2 + B` to enter the robot's damping/zero-torque safe state. While in that
   state, press `L2 + R2` on the controller to enter debug mode.

5. **Run the deployment binary**:
   ```bash
   cd deploy/robots/go2/build
   ./go2_ctrl --network=network_name # Found via ifconfig
   ```

   To run one of several exported policy directories, add `--policy_dir`:
   ```bash
   ./go2_ctrl \
       --network=network_name \
       --policy_dir=config/policy/velocity/policies/<policy_name>
   ```

   At startup, the binary prompts for velocity command input:
   ```text
   Select velocity command input ([c]ontroller / [k]eyboard):
   ```
   Choose `c` to use the controller sticks for velocity commands. Choose `k`
   to use keyboard velocity commands; in keyboard mode, `Up` / `Down` adjust
   `vx` in 0.1 increments and `R` resets `vx` to zero. The controller is still
   required for FSM transitions and passive aborts.

   Watch the startup log closely. It should report that no Unitree high-level
   motion service is active before waiting for `LowState`. If you see:
   ```text
   Another process is still publishing on rt/lowcmd after releasing Unitree's motion service.
   ```
   the binary has refused to start because a non-Unitree low-level controller is
   still publishing. Stop that process and relaunch before entering `FixStand`
   or `Velocity`.

6. **Operate the FSM** via the controller in either input mode:
   - `L2 + Up` -- transition from Passive -> FixStand (robot stands up slowly)
   - `R2 + A` -- transition from FixStand -> Velocity (policy takes over)
   - `L2 + B` -- return to Passive at any point (safe abort)

   Note: this repo's `Passive` state is a damping state, not true zero torque.
   It sets `kp = 0`, `kd = 3`, and continuously publishes each motor's current
   position, so some viscous limb resistance is expected. Rattling, rapid
   oscillation, or repeated `lowcmd` warnings are not expected; stop, re-check
   that `sport_mode`/`ai_sport`/`advanced_sport` is off, and relaunch cleanly.

---

## Notes
NOTE: The policies we want to deploy is the following

```
quadruped-velocity_tracking-SAC-20260305-042406
```

which was trained with the following settings:
```json
{
  "algorithm": "SAC",
  "learning_rate": 0.0003,
  "buffer_size": 1000000,
  "learning_starts": 50000,
  "batch_size": 256,
  "tau": 0.005,
  "gamma": 0.99,
  "gradient_steps": -1,
  "seed": 1,
  "tensorboard_log": "logs/quadruped-velocity_tracking-SAC-20260305-042406/tensorboard",
  "inject_n_timesteps": 5000,
  "inject_type": "percentage",
  "percentage": 25,
  "num_traj": 10,
  "random_select": true,
  "data_dir": null,
  "env_name": "quadruped-velocity_tracking",
  "domain": "quadruped",
  "task": "velocity_tracking",
  "total_timesteps": 3000000,
  "num_envs": 512,
  "save_replay_buffer_checkpoints": false,
  "save_replay_buffer_final": false,
  "domain_randomization": {
    "enabled": true,
    "obs_noise_level": 1.0
  }
}
```

and

```
quadruped-velocity_tracking-SAC-20260305-170808
```

which was trained with the following settings:
```json
{
  "algorithm": "SAC",
  "learning_rate": 0.0003,
  "buffer_size": 1000000,
  "learning_starts": 50000,
  "batch_size": 256,
  "tau": 0.005,
  "gamma": 0.99,
  "gradient_steps": -1,
  "seed": 1,
  "tensorboard_log": "logs/quadruped-velocity_tracking-SAC-20260305-170808/tensorboard",
  "inject_n_timesteps": 5000,
  "inject_type": "percentage",
  "percentage": 25,
  "num_traj": 10,
  "random_select": true,
  "data_dir": null,
  "env_name": "quadruped-velocity_tracking",
  "domain": "quadruped",
  "task": "velocity_tracking",
  "total_timesteps": 5000000,
  "num_envs": 512,
  "save_replay_buffer_checkpoints": false,
  "save_replay_buffer_final": false,
  "domain_randomization": {
    "enabled": true,
    "obs_noise_level": 1.0
  }
}
```
and command:
```
python mpc_rl/train.py --env_name quadruped-velocity_tracking --algorithm SAC --total_timesteps 5000000 --num_envs 512 --seed 1 --learning_starts 50000 --save_replay_buffer_checkpoints=False --save_replay_buffer_final=False --domain_rand=True
```
