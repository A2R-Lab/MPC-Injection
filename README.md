# MPC-Injection Experimental Workspace
This repo is a test space for research ideas in combining MPC and RL methods to leverage both their strengths for robot control.

The main training script (`mpc_rl/train.py`) provides a flexible command-line interface for training, evaluation, checkpointing, MPC trajectory injection, and video recording. It supports dm_control tasks through SBX (Stable Baselines Jax), SB3-based quadruped policies, SAC-MPC / TD3-MPC variants, cheetah3 experiments, shadow hand experiments, and Go2 export/deployment workflows.

## Prerequisites
- Conda (recommended) or Python virtual environment
- CUDA-capable GPU (optional but recommended for faster training)
- Python 3.11

## Installation

Create the environment from the `environment.yml` file:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate mpc-rl
```

### Installing the mpc_rl Package

Install the `mpc_rl` package in editable mode so that Python can import it from anywhere. This is required for running tests and scripts that import `mpc_rl` modules.

From the project root directory, run:

```bash
pip install -e .
```

This installs the package in editable mode, meaning any changes you make to the source code will be immediately available without reinstalling. You can verify the installation worked by running:

```bash
python -c "import mpc_rl; import mpc_rl.envs; print('mpc_rl successfully installed!')"
```

### Installing MuJoCo-MPC
Note that this project is using Ubuntu 24.04.3 LTS and clang version 18.1.3. It should be possible to install mjpc through the terminal command seen in the mjpc submodule's README. However, it is so much easier to let VSCode compile the project for you. The process is copied over from mjpc's README and shown here for convenience.

#### Build and Run MJPC GUI application using VSCode
We recommend using [VSCode](https://code.visualstudio.com/) and 2 of its extensions ([CMake Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cmake-tools) and [C/C++](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools)) to simplify the build process.

1. Open the submodule directory `deps/mujoco_mpc` with VSCode.
2. Configure the project with CMake (a pop-up should appear in VSCode)
3. Set compiler to `clang-18`.
4. Build `[all]` and then you can test by running `./mjpc` in `mujoco_mpc/build/bin` or try the next step.
5. Build and run the `mjpc` target in "release" mode (VSCode defaults to "debug"). This will open and run the graphical user interface.

### Installing MuJoCo-MPC Python Bindings
Make sure you're doing this in the mpc-rl conda environment and after you've built the project as seen above. Next, change to mujoco_mpc's python directory (or else the script can't find certain file paths):
```bash
cd deps/mujoco_mpc/python
```

Install the Python module:
```bash
python setup.py install
```

Test that installation was successful by going back to the root project directory and trying:
```bash
python deps/mujoco_mpc/python/mujoco_mpc/agent_test.py
```
This should result in 18 tests run with 1 failed and 3 skipped. This is okay for now.

Example scripts are found in `deps/mujoco_mpc/python/mujoco_mpc/demos`. For example, from the project root:
```bash
python deps/mujoco_mpc/python/mujoco_mpc/demos/agent/cartpole_gui.py
```
will run the MJPC GUI application using MuJoCo's passive viewer via Python.

### Installing gym-quadruped
The gym-quadruped package provides quadruped robot environments. Install it locally from the submodule:

```bash
pip install -e deps/gym-quadruped
```

### Installing mpx
The mpx library provides GPU-accelerated MPC and trajectory optimization in JAX for legged robots. Install it locally from the submodule:

```bash
pip install -e deps/mpx
```

Test that mpx is working by running an example:
```bash
python deps/mpx/mpx/examples/mjx_quad.py
```
Note: The first time running the script may take over a minute to JIT the solver. Use keyboard arrows to control the robot.

TODO: Test mpc-rl's trajectory collector here
```bash
python ...
```

## Usage

### Training a New Agent

Train an SAC agent on a dm_control environment:

```bash
python mpc_rl/train.py --env_name=cartpole-swingup
```

### Training a Quadruped Policy

To train a quadruped velocity-tracking policy use:

```bash
python mpc_rl/train.py --env_name quadruped-velocity_tracking --algorithm SAC --total_timesteps 5000000 --num_envs 512 --seed 1 --learning_starts 50000 --save_replay_buffer_checkpoints=False --save_replay_buffer_final=False --domain_rand=True
```

Quadruped environments retain the existing
`residual_position_scale_0p5_lpf_5hz_v1` action interface by default. The
MPX bounding-data path can explicitly select
`--quadruped_action_interface=mpx_bound_scale_1_no_lpf_v1`; this uses a
1-radian residual scale with no action-target LPF. Schema-v2 direct-transition
loading validates that the dataset and training interface match. The accepted
transition-parity result is documented in `docs/mpx_transition_parity.md`.
Passing this transition gate does not by itself approve bulk bound-data
generation or training.

Test a trained quadruped model locally with:

```bash
python mpc_rl/play_quad.py --model=logs/quadruped-velocity_tracking-*-*-*
```

The retained G6-G8 Go2 barrel-roll route uses `quadruped-barrel_roll`. It is
SAC-MPC only and requires schema-v2 direct replay at 25%, Go2, disabled domain
randomization, enabled Go2 sysID, `action_scale=2.0`, and the task's fixed
70-step horizon. The historical schema-v1/`action_scale=0.5` G6 pilot is
**blocked**, not passed: held-out success remained 0% while every demonstration
clipped more than 20% of its saved inverse-PD residual actions. Schema v2 is the
controlled representability recovery and must not be mixed with v1. Its 100k
G6 retry passed the learning-progress gate by reaching a reproducible 15%
held-out success at the success-selected 80k checkpoint, although the final
policy regressed to 0%; checkpoint instability remains an explicit production
risk. G7 then passed at root `6fdd5bee` and promoted the immutable production
dataset to `data/go2_barrel_roll/v2`: 1,000 unique schema-v2 files, 70,000
direct transitions, and 1,435 retained attempts (1,000 accepted and 435
rejected). Mean action clipping is 0.4987% and the worst file is 1.7857%.
The effective configuration SHA-256 is
`8013d45f6ea6540242a0aa49f5cf728d736e08e451ae498edb697703da803c0d`,
the aggregate-manifest SHA-256 is
`8d67b8ab296bc4bd56cffac69e34a44f2b8da922b203e0b0b9c1b4ad931b5491`,
and the checksum-index SHA-256 is
`153334c544fec09e8aee4fa74223bde0f5b1c6b4f0018ce1e65328fbf0ccee70`.
Verify the promoted artifact with:

```bash
conda run --no-capture-output -n mpc-rl python \
  mpc_rl/planner/barrel_roll_dataset.py data/go2_barrel_roll/v2

(
  cd data/go2_barrel_roll/v2 &&
  sha256sum --check --strict checksums.sha256
)
```

G8 production training is **BLOCKED**. Predeclared seeds 1 and 2 each completed
500,000 steps, but their success-selected held-out rates were only 1%. Because
even a hypothetical 100% seed 3 would leave the median at 1%, the project owner
authorized stopping before seed 3. Do not rerun or overwrite the retained
campaign under `logs/go2_barrel_roll_g8_production/`.

G9 schema-v3 recovery is **complete**. It keeps the 1.40 s MPC maneuver but
extends each episode and accepted trajectory to 2.50 s: 125 policy steps at
50 Hz and 500 MuJoCo/PD steps at 200 Hz. After 1.40 s, generation repeats the
terminal-padded final-stance plan without further MPC replanning. Schema v3
also uses the locked roll/rate/action-change reward and a final-only classifier
based on roll progress, base height, and body-up tilt. It preserves the 45D
actor, 4D privileged critic, 5 Hz online target LPF, direct-torque demonstration
execution, inverse-PD saved labels, 25% direct injection, disabled DR, and Go2
sysID.

All G9.1 through G9.6 gates passed. The immutable promoted dataset at
`data/go2_barrel_roll/v3` contains
exactly 1,000 accepted trajectories and 125,000 direct transitions. The fixed
data runner now refuses to overwrite staging, v3, or its audit:

```bash
./run_go2_barrel_roll_g9_dataset.sh
```

After promotion, validate both the schema and content hashes with:

```bash
conda run --no-capture-output -n mpc-rl python \
  mpc_rl/planner/barrel_roll_dataset.py data/go2_barrel_roll/v3

(
  cd data/go2_barrel_roll/v3 &&
  sha256sum --check --strict checksums.sha256
)
```

The G9 campaign runner has separate concurrent `smoke` and `production` modes.
It hard-checks the promoted dataset identity, refuses existing campaigns, runs
training seeds 1 and 2 together, samples resources, disables replay-buffer
saves, and terminates the peer process if either job fails:

```bash
./run_go2_barrel_roll_g9_campaign.sh smoke
./run_go2_barrel_roll_g9_campaign.sh production
```

Production selects the earliest checkpoint reaching each seed's best rate over
validation seeds `2000000` through `2000099`. It then locks both selections and
scores each exactly once on final-test seeds `3000000` through `3000099` via
`mpc_rl/evaluate_go2_barrel_roll_g9.py`. Seed 1 selected step 220,000 and scored
100/100; seed 2 selected step 230,000 and scored 98/100. The official seed-1
model therefore exceeds the 80/100 acceptance threshold. Exact gate evidence,
artifact hashes, resource measurements, and video review are maintained in
`docs/go2_barrel_roll_mpc_injection_plan.md`.

Schema-v4 0% training uses the audited
`mpc_rl/run_go2_barrel_roll_v4_0pct.py` launcher. The launcher preserves the
hash-locked v4 training source, requires the same immutable v4 dataset and
frozen hyperparameters, and writes `COMPLETE` only if the observed MPC replay
percentage stayed exactly 0 throughout training. The normal v4 production
route remains fixed at 25%.

Train seed 1 of the 0% baseline with:

```bash
conda run --no-capture-output -n mpc-rl python \
  mpc_rl/run_go2_barrel_roll_v4_0pct.py \
  --env_name=quadruped-barrel_roll \
  --algorithm=SAC-MPC \
  --robot=go2 \
  --use_go2_sysid=true \
  --seed=1 \
  --total_timesteps=500000 \
  --num_envs=4 \
  --max_episode_steps=125 \
  --logdir=logs/go2_barrel_roll_v4_0pct/seed1 \
  --suffix=v4-0pct-seed1 \
  --enable_logging=true \
  --learning_rate=0.0003 \
  --buffer_size=1000000 \
  --learning_starts=10000 \
  --batch_size=256 \
  --tau=0.005 \
  --gamma=0.99 \
  --gradient_steps=-1 \
  --policy_delay=2 \
  --inject_type=percentage \
  --percentage=0 \
  --random_select=true \
  --data_dir=data/go2_barrel_roll/v4 \
  --quadruped_mpc_replay_mode=direct \
  --checkpoint_freq=25000 \
  --eval_freq=10000 \
  --save_replay_buffer_checkpoints=false \
  --save_replay_buffer_final=false \
  --domain_rand=false \
  --domain_rand_config_type=disabled
```

After it completes, render ten non-scored rollouts from the selected best
checkpoint, record one of those seeds as a Viser trajectory, and launch Viser:

```bash
conda run --no-capture-output -n mpc-rl python \
  mpc_rl/evaluate_go2_barrel_roll_v4_0pct.py \
  --run-dir=logs/go2_barrel_roll_v4_0pct/seed1

conda run --no-capture-output -n mpc-rl python \
  body_trajs/record_go2_barrel_roll_policy.py \
  --run-dir=logs/go2_barrel_roll_v4_0pct/seed1 \
  --seed=7000000 \
  --output-dir=body_trajs/model_traj_data_quadruped/v4_0pct

conda run --no-capture-output -n mpc-rl python \
  viser/viser_quadruped_viz_trajs.py \
  --trajectory=/absolute/path/printed/by/the/recorder.npz \
  --port=8081
```

The video summary lists `successful_seeds`; substitute one of them for
`7000000` and add `--require-success` to the recorder when a successful
trajectory is required. The default video seeds `7000000` through `7000009`
are disjoint from v4 validation (`2000000`-`2000099`) and sealed final-test
(`8000000`-`8000099`) seeds.

The frozen sequential runner revalidates the immutable dataset, records
provenance and resource usage, refuses to overwrite an existing campaign, and
stops on the first failed run:

```bash
./run_go2_barrel_roll_sac_mpc.sh
```

It writes ignored policies, checkpoints, evaluation JSONL, selected-checkpoint
reports, and labelled representative outcome videos under the campaign
directory. The non-training production routing check is:

```bash
python mpc_rl/train.py \
    --env_name=quadruped-barrel_roll \
    --robot=go2 \
    --algorithm=SAC-MPC \
    --total_timesteps=0 \
    --num_envs=1 \
    --enable_logging=False \
    --inject_type=percentage \
    --percentage=25 \
    --quadruped_mpc_replay_mode=direct \
    --data_dir=data/go2_barrel_roll/v2 \
    --domain_rand=False \
    --domain_rand_config_type=disabled \
    --use_go2_sysid=True
```

The retained G8 checkpoints were selected over seeds `1000000` through
`1000099`. G9 uses the disjoint ranges documented above. Barrel videos are
seed-labelled and do not set or display velocity commands.

## Deploy

See `deploy/README.md` for Go2 ONNX export, simulation testing, C++ build, and real-robot deployment instructions.

The `--env_name` flag uses the `domain-task` format. Underscores can appear inside the task name, as in `quadruped-velocity_tracking`, but the separator between domain and task is a hyphen. You can quickly list dm_control environments with this command:
```
python -c "from dm_control import suite; print('\n'.join([f'{domain}/{task}' for domain, task in suite.ALL_TASKS]))"
```

**Common Training Options:**
```bash
# Train with custom hyperparameters
python mpc_rl/train.py \
    --env_name=cartpole-swingup \
    --algorithm=SAC \
    --total_timesteps=200000 \
    --num_envs=8 \
    --learning_rate=1e-3 \
    --batch_size=512

# Train with a different algorithm (PPO or TD3)
python mpc_rl/train.py --env_name=acrobot-swingup --algorithm=TD3

# Train with SAC-MPC
python mpc_rl/train.py \
    --env_name=cartpole-swingup \
    --algorithm=SAC-MPC \
    --total_timesteps=500000 \
    --inject_n_timesteps=5000 \
    --num_traj=10 \
    --random_select=True \
    --data_dir=data/cartpole_0_001dt/

# Add a custom suffix to the experiment name
python mpc_rl/train.py --env_name=cartpole-swingup --suffix=experiment1
```

### Evaluating a Trained Agent

Evaluate a trained model and generate videos:

```bash
python mpc_rl/train.py \
    --env_name=cartpole-swingup \
    --play_only \
    --load_run_name=cartpole-swingup-SAC-20251006-143022
```

This will:
- Load the trained model from the specified run
- Evaluate for 5 episodes (default)
- Record 3 videos as MP4 files in the `videos/` subdirectory
- Print performance statistics

**Evaluation Options:**
```bash
# Evaluate with more episodes and videos
python mpc_rl/train.py \
    --env_name=cartpole-swingup \
    --play_only \
    --load_run_name=cartpole-swingup-SAC-20251006-143022 \
    --num_eval_episodes=10 \
    --num_videos=5
```

### Continuing Training from a Checkpoint

Resume training from a previously saved checkpoint:

```bash
python mpc_rl/train.py \
    --env_name=cartpole-swingup \
    --load_run_name=cartpole-swingup-SAC-20251006-143022 \
    --total_timesteps=200000
```

This will load the existing model and normalization statistics, then continue training for the specified number of additional timesteps.

## Command-Line Flags

### Environment Flags
- `--env_name`: Environment name in `domain-task` format (e.g., `cartpole-swingup`, `acrobot-swingup`, `walker-walk`, `quadruped-velocity_tracking`)
- `--domain`: Domain name. Optional; parsed from `--env_name` if not provided.
- `--task`: Task name. Optional; parsed from `--env_name` if not provided.
- `--robot`: Quadruped robot model for quadruped environments. Default: `go2`
- `--use_go2_sysid`: Apply the identified Go2 joint-dynamics patch for quadruped environments and MPX controllers. Default: `True`
- `--quadruped_action_interface`: Versioned quadruped action interface. Default: `residual_position_scale_0p5_lpf_5hz_v1`; opt-in MPX bound interface: `mpx_bound_scale_1_no_lpf_v1`
- `--max_episode_steps`: Maximum episode length. Default: `1000`
- `--cheetah3_speed_goal`: Forward speed target for cheetah3 reward. Default: `3.0`

### Training Flags
- `--algorithm`: RL algorithm to use (`SAC`, `PPO`, `TD3`, `SAC-MPC`, or `TD3-MPC`). Default: `SAC`
- `--total_timesteps`: Total number of training timesteps. Default: `500000`
- `--num_envs`: Number of parallel environments for training. Default: `4`
- `--seed`: Random seed for reproducibility. Default: `1`

### Evaluation Flags
- `--play_only`: Skip training and only evaluate the model. Default: `False`
- `--load_run_name`: Name of the run directory to load from (e.g., `cartpole-swingup-SAC-20251006-143022`)
- `--num_eval_episodes`: Number of episodes to run during evaluation. Default: `5`
- `--num_videos`: Number of videos to record during evaluation. Default: `3`
- `--checkpoint_evals`: Comma-separated checkpoint steps for additional video evaluations (e.g., `400000,500000`)

### Experiment Flags
- `--suffix`: Custom suffix to append to the experiment name
- `--logdir`: Base directory for logs, checkpoints, videos, and TensorBoard output. Default: `logs`
- `--enable_logging`: Enables checkpoints, videos, and TensorBoard logging. Set to `False` for hyperparameter optimization. Default: `True`

### Hyperparameter Flags (SAC/TD3 variants)
- `--learning_rate`: Learning rate. Default: `3e-4`
- `--buffer_size`: Replay buffer size. Default: `1000000`
- `--learning_starts`: Steps to collect transitions before learning starts. Default: `10000`
- `--policy_delay`: TD3 actor update delay. Default: `2`
- `--batch_size`: Minibatch size. Default: `256`
- `--tau`: Target network update rate. Default: `0.005`
- `--gamma`: Discount factor. Default: `0.99`
- `--gradient_steps`: Gradient steps per rollout. Default: `-1`

### MPC Injection Flags (SAC-MPC / TD3-MPC only)
- `--inject_n_timesteps`: Inject MPC trajectories every N timesteps for fixed injection. Default: `5000`
- `--inject_type`: MPC injection mode (`percentage` or `fixed`). Default: `percentage`
- `--percentage`: Target replay-buffer percentage for MPC trajectories. Default: `25`
- `--num_traj`: Number of fixed MPC trajectories to inject each time. Default: `10`
- `--random_select`: Randomly select trajectories to inject. Default: `True`
- `--data_dir`: Directory containing pre-generated MPC trajectories. Default: `None`

### Checkpoint and Replay-Buffer Flags
- `--checkpoint_freq`: Save checkpoint every N environment steps. Default: `25000`
- `--eval_freq`: Evaluate policy every N environment steps during training. Default: `10000`
- `--save_replay_buffer_checkpoints`: Save replay buffer at each checkpoint. Default: `False`
- `--save_replay_buffer_final`: Save replay buffer at the end of training for resume support. Default: `False`

### Domain-Randomization Flags
- `--domain_rand`: Enable quadruped domain randomization. Default: `True`
- `--domain_rand_config_type`: Named quadruped domain-randomization preset. Default: `custom`
- `--domain_rand_obs_noise`: Observation noise level for custom domain randomization. Default: `1.0`

## Directory Structure
This is the current high-level repo layout for training, MPC data generation, rollout recording, and plotting.
```
MPC-Injection/
├── body_trajs/                      # Rollout recording + trajectory plotting utilities
│   ├── model_traj_data_quadruped/   # Saved quadruped rollouts from record_body_trajs_by_policy_quadruped.py
│   ├── model_traj_data_walker/      # Saved walker rollouts from record_body_trajs_by_policy_walker.py
│   ├── plot_body_trajs_by_policy_walker.py
│   ├── plot_foot_trajectory_quadruped.py
│   ├── plot_torque_trajs_quadruped.py
│   ├── plot_tsne_state_distr_by_policy_walker.py
│   ├── plot_umap_state_distr_by_policy_walker.py
│   ├── plot_box_body_trajs_by_policy_walker.py
│   ├── plot_torque_cdf_quadruped.py
│   ├── record_body_trajs_by_policy_quadruped.py
│   └── record_body_trajs_by_policy_walker.py
├── data/                            # MPC trajectory datasets consumed by SAC-MPC / TD3-MPC runs
│   ├── cartpole_0_001dt/
│   ├── cartpole_0_001dt_all/
│   ├── cheetah3_0_010dt/
│   ├── quadruped/
│   ├── quadruped_dr/
│   ├── quadruped_full_vel_cmds/
│   ├── quadruped_old/
│   ├── quadruped_old_old/
│   ├── shadow_hand_0_002dt/
│   ├── shadow_hand_0_002dt_test/
│   └── walker_0_0025dt/
├── deploy/                          # ONNX export / deployment experiments
│   ├── export_onnx_go2.py
│   ├── test_onnx_policy.py
│   ├── robots/
│   ├── thirdparty/
│   └── README.md
├── deps/                            # External dependencies and submodules
│   ├── gym-quadruped/              # Quadruped robot environments (pip install -e)
│   ├── mpx/                        # GPU-accelerated MPC in JAX (pip install -e)
│   └── mujoco_mpc/                 # MuJoCo MPC library
├── environment.yml
├── jax_cache/                       # Local JAX compilation cache
├── LICENSE
├── logs/                            # Saved policies, checkpoints, eval videos, tensorboard logs
│   ├── quadruped-velocity_tracking-*/
│   ├── quadruped_domain_rand*/
│   ├── SAC-MPC-quadruped*/
│   ├── TD3-MPC-quadruped*/
│   ├── SAC-MPC-walker*/
│   └── TD3-MPC-walker*/
├── mpc_rl/
│   ├── asym_policies/               # Asymmetric policy definitions used by quadruped training
│   ├── common/                      # Replay buffers, callbacks, and shared utilities
│   ├── envs/                        # Environment registration, wrappers, DR config
│   ├── planner/                     # MPC planners + trajectory data generation scripts
│   ├── sac_mpc/                     # SAC-MPC implementation
│   ├── tasks/                       # Task assets and task-specific helpers
│   ├── td3_mpc/                     # TD3-MPC implementation
│   ├── play_quad.py                 # Local quadruped playback helper
│   └── train.py                     # Main training / eval entrypoint
├── plot_over_training_reward_surfaces.sh  # Bash pipeline to plot reward surfaces over checkpoints
├── plots/                           # Output figures from body_trajs/, utils/, viser/, reward_surfaces/
│   └── body_trajectory_plots/
├── plot_single_reward_surface.sh    # Bash pipeline to plot one reward surface for a saved model
├── README.md
├── reward_surfaces/                 # Reward-surface package + scripts
│   ├── reward_surfaces/
│   │   ├── core/                    # Evaluation and surface generation logic
│   │   ├── plotting/                # Matplotlib plotting helpers
│   │   └── utils/
│   ├── scripts/                     # CLI scripts for job generation, evaluation, aggregation, plotting
│   └── README.md
├── run_quadruped_domain_rand.sh     # Main quadruped domain-randomization experiment sweep
├── run_quadruped_experiments.sh     # Quadruped MPC-percentage experiment sweep
├── run_quadruped_rwrd_shpng_sac.sh  # Quadruped SAC reward-shaping experiment sweep
├── run_quadruped_rwrd_shpng_td3.sh  # Quadruped TD3 reward-shaping experiment sweep
├── run_cheetah_experiments.sh       # Cheetah3 experiment sweep
├── run_walker_experiments.sh        # Walker MPC-percentage experiment sweep
├── setup.py
├── tests/                           # Diagnostics, environment checks, and trajectory-generation tests
├── utils/
│   ├── check_data_integrity.py
│   ├── plot_train_pct_comparisons_quadruped.py
│   └── plot_train_pct_comparisons_walker.py
└── viser/
    ├── cnvrt_mjcf_to_urdf.py
    ├── plot_footsteps_n_rewards_quadruped.py
    ├── plot_footsteps_n_rewards_walker.py
    ├── plot_footsteps_walker.py
    ├── viser_figs/                  # Saved visualization figures
    ├── viser_quadruped_viz_trajs.py
    ├── viser_tutorials.py
    └── viser_walker_viz_trajs.py
```

Plotting entry points currently used in this repo:

- Python plotting scripts:
  - `body_trajs/plot_foot_trajectory_quadruped.py`
  - `body_trajs/plot_torque_trajs_quadruped.py`
  - `body_trajs/plot_body_trajs_by_policy_walker.py`
  - `body_trajs/plot_tsne_state_distr_by_policy_walker.py`
  - `body_trajs/plot_umap_state_distr_by_policy_walker.py`
  - `body_trajs/plot_box_body_trajs_by_policy_walker.py`
  - `body_trajs/plot_torque_cdf_quadruped.py`
  - `utils/plot_train_pct_comparisons_quadruped.py`
  - `utils/plot_train_pct_comparisons_walker.py`
  - `viser/plot_footsteps_n_rewards_quadruped.py`
  - `viser/plot_footsteps_n_rewards_walker.py`
  - `viser/plot_footsteps_walker.py`
  - `reward_surfaces/scripts/plot_plane.py`
  - `reward_surfaces/scripts/replot_from_data.py`
- Bash plotting pipelines:
  - `plot_single_reward_surface.sh`
  - `plot_over_training_reward_surfaces.sh`

Related visualization / data-prep scripts:

- `body_trajs/record_body_trajs_by_policy_quadruped.py` and `body_trajs/record_body_trajs_by_policy_walker.py` generate rollout `.npz` files used by the plotting scripts above.
- `viser/viser_quadruped_viz_trajs.py` and `viser/viser_walker_viz_trajs.py` are interactive visualization tools for recorded trajectories rather than static plotting scripts.


After running training, the logs are organized as follows:

```
logs/
└── {env_name}-{algorithm}-{timestamp}[-injection][-suffix]/
    ├── config.json                    # Training configuration
    ├── final_model.zip                # Final trained model
    ├── vec_normalize.pkl              # Observation normalization statistics
    ├── checkpoints/                   # Periodic checkpoints during training
    │   ├── model_25000_steps.zip
    │   ├── model_50000_steps.zip
    │   ├── model_vecnormalize_25000_steps.pkl
    │   ├── model_replay_buffer_25000_steps.pkl  # Only when replay-buffer checkpoints are enabled
    │   └── ...
    ├── best_model/                    # Best model based on evaluation
    │   └── best_model.zip
    ├── eval_logs/                     # Evaluation metrics during training
    │   └── evaluations.npz
    ├── videos/                        # Evaluation videos (MP4)
    │   ├── rollout0.mp4
    │   ├── rollout1.mp4
    │   └── rollout2.mp4
    └── tensorboard/                   # TensorBoard logs
        └── {algorithm}_1/
```

### Key Files

- **`final_model.zip`**: The trained model after completing all timesteps
- **`vec_normalize.pkl`**: Statistics for observation/reward normalization (important for evaluation)
- **`config.json`**: Records all hyperparameters and settings used for training
- **`checkpoints/`**: Contains periodic snapshots of the model during training
- **`checkpoints/model_vecnormalize_*_steps.pkl`**: VecNormalize snapshots paired with checkpointed models
- **`checkpoints/model_replay_buffer_*_steps.pkl`**: Replay-buffer snapshots, if `--save_replay_buffer_checkpoints=True`
- **`best_model/`**: Contains the best-performing model based on evaluation metrics
- **`videos/`**: MP4 recordings of the agent's behavior during evaluation

## Monitoring Training

TensorBoard logs are automatically saved during training. To monitor progress:

```bash
tensorboard --logdir logs/{your_run_name}/tensorboard
```

Then open your browser to `http://localhost:6006` to view training metrics in real-time.

## Troubleshooting

### DMControl 2D Walker Reward Function Change
If you installed the DMControl environments via pip then to run the experiments with a simple (velocity only) reward function you need to go into the `site-packages/dm_control/suite/walker.py` installation in your conda environment and edit the line `return stand_reward * (5*move_reward + 1) / 6` to `return (5*move_reward + 1) / 6`

### Running Headless
If you're running headless and want to save videos make sure to set these environmental variables
```
export MUJOCO_GL=egl
unset DISPLAY
```

### JAX GPU Recognition Issue
There is a known bug where JAX may not recognize the GPU. A temporary fix:

```bash
sudo rmmod nvidia_uvm
sudo modprobe nvidia_uvm
```

See this [issue discussion](https://github.com/jax-ml/jax/issues/28980) for more details.

### Model Not Loading
Ensure you're using the exact run name (with timestamp) when loading checkpoints. Check the `logs/` directory for available runs.
