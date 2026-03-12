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
  Linear(45 -> 256) -> ReLU -> Linear(256 -> 256) -> ReLU -> Linear(256 -> 12) -> Tanh()
  ```
  There is **no** `latent_pi` attribute. The full TD3 inference is:
  `normalise -> mu` (tanh is the last layer inside `mu`)

Both produce actions in `[-1, 1]`. The C++ action manager
(`JointPositionAction` in `deploy/include/isaaclab/envs/mdp/actions/joint_actions.h`)
is the component that converts raw neural network outputs into physical joint
position targets. It reads `scale` and `offset` arrays from `deploy.yaml` and
applies them element-wise to each action dimension. For the Go2, this becomes:
```
joint_target[i] = action[i] * 0.5 + default_joint_pos[i]
```
identically for both algorithms. The processed targets are then written to the
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
  -> mu:        Linear(45->256) -> ReLU -> Linear(256->256) -> ReLU -> Linear(256->12) -> Tanh
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
   d. Scales: `joint_target = action * 0.5 + default_joint_pos`
   e. Writes position targets to `LowCmd` which the SDK sends to the motors
4. PD gains (stiffness / damping) are set from `config/policy/velocity/v0/params/deploy.yaml`.

### Observation vector layout (45 dimensions)

| Slice | Name | Source | Dim |
|-------|------|--------|-----|
| `[0:3]`   | `base_ang_vel`    | IMU gyroscope (body frame) | 3 |
| `[3:6]`   | `projected_gravity` | IMU orientation -> R^T*[0,0,-1] | 3 |
| `[6:9]`   | `velocity_commands` | Joystick (vx, vy, wz) | 3 |
| `[9:21]`  | `joint_pos_rel`   | Encoder pos - default pos | 12 |
| `[21:33]` | `joint_vel_rel`   | Encoder velocities | 12 |
| `[33:45]` | `last_action`     | Previous policy output | 12 |

### Joint ordering

The policy uses joints in the order **FR -> FL -> RR -> RL** (front-right,
front-left, rear-right, rear-left), each with hip -> thigh -> calf.

The Unitree SDK2 numbers motor indices differently (0-2 = FL, 3-5 = FR,
6-8 = RL, 9-11 = RR).  The mapping in `deploy.yaml` is:

```yaml
joint_ids_map: [3,4,5, 0,1,2, 9,10,11, 6,7,8]
# policy idx:   FR       FL     RR        RL
# SDK2 motor:   FR       FL     RR        RL  (re-indexed)
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

### Step B -- Test the ONNX policy in simulation

Before touching the real robot, verify the ONNX policy drives the simulated
Go2 correctly using the same MuJoCo environment it was trained in:

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
   - Address: `192.168.123.99`
   - Netmask: `255.255.255.0`
   Use `ifconfig` to find the Ethernet interface name (e.g. `enp5s0`).

2. **Power on** the robot in a safe state (use a harness).

3. **Enter zero-torque & then debug mode**: Press `L2 + B` to enter zero-torque mode. While in zero-torque mode, press `L2 + R2` on the controller to enter debug mode.

4. **Run the deployment binary**:
   ```bash
   cd deploy/robots/go2/build
   ./go2_ctrl --network=network_name # Found via ifconfig
   ```

5. **Operate the FSM** via the controller:
   - `L2 + Up` -- transition from Passive -> FixStand (robot stands up slowly)
   - `R2 + A` -- transition from FixStand -> Velocity (policy takes over)
   - `L2 + B` -- return to Passive at any point (safe abort)

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
