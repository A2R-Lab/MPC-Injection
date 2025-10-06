# MPC-RL Experimental Workspace
This repo is a test space for research ideas in combining MPC and RL methods to leverage both their strengths for robot control.

The initial setup uses dm_control environments with SBX (Stable Baselines Jax) for training RL policies. The main training script (`train_sbx.py`) provides a flexible command-line interface for training, evaluation, and checkpoint management.

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

### Installing MuJoCo-MPC
TODO: go over the details of installing MuJoCo-MPC locally and the python install. Testing this out in a new branch to be safe!

## Usage

### Training a New Agent

Train an SAC agent on a dm_control environment:

```bash
python mpc_rl/train_sbx.py --env_name=cartpole-swingup
```

Note that we keep the `--env_name` flag to be split up into `domain-task` or `domain_task`. You can quickly list the available environments with this command:
```
python -c "from dm_control import suite; print('\n'.join([f'{domain}/{task}' for domain, task in suite.ALL_TASKS]))"
```

**Common Training Options:**
```bash
# Train with custom hyperparameters
python mpc_rl/train_sbx.py \
    --env_name=cartpole-swingup \
    --algorithm=SAC \
    --total_timesteps=200000 \
    --num_envs=8 \
    --learning_rate=1e-3 \
    --batch_size=512

# Train with a different algorithm (PPO or TD3)
python mpc_rl/train_sbx.py --env_name=acrobot-swingup --algorithm=TD3

# Add a custom suffix to the experiment name
python mpc_rl/train_sbx.py --env_name=cartpole-swingup --suffix=experiment1
```

### Evaluating a Trained Agent

Evaluate a trained model and generate videos:

```bash
python mpc_rl/train_sbx.py \
    --env_name=cartpole-swingup \
    --play_only \
    --load_run_name=cartpole-swingup-20251006-143022
```

This will:
- Load the trained model from the specified run
- Evaluate for 5 episodes (default)
- Record 3 videos as MP4 files in the `videos/` subdirectory
- Print performance statistics

**Evaluation Options:**
```bash
# Evaluate with more episodes and videos
python mpc_rl/train_sbx.py \
    --env_name=cartpole-swingup \
    --play_only \
    --load_run_name=cartpole-swingup-20251006-143022 \
    --num_eval_episodes=10 \
    --num_videos=5
```

### Continuing Training from a Checkpoint

Resume training from a previously saved checkpoint:

```bash
python mpc_rl/train_sbx.py \
    --env_name=cartpole-swingup \
    --load_run_name=cartpole-swingup-20251006-143022 \
    --total_timesteps=200000
```

This will load the existing model and normalization statistics, then continue training for the specified number of additional timesteps.

## Command-Line Flags

### Environment Flags
- `--env_name`: Environment name in format `domain-task` (e.g., `cartpole-swingup`, `acrobot-swingup`, `walker-walk`)
- `--domain`: Domain name (optional, parsed from env_name if not provided)
- `--task`: Task name (optional, parsed from env_name if not provided)

### Training Flags
- `--algorithm`: RL algorithm to use (`SAC`, `PPO`, or `TD3`). Default: `SAC`
- `--total_timesteps`: Total number of training timesteps. Default: `100000`
- `--num_envs`: Number of parallel environments for training. Default: `4`
- `--seed`: Random seed for reproducibility. Default: `1`

### Evaluation Flags
- `--play_only`: Skip training and only evaluate the model. Default: `False`
- `--load_run_name`: Name of the run directory to load checkpoint from (e.g., `cartpole-swingup-20251006-143022`)
- `--num_eval_episodes`: Number of episodes to run during evaluation. Default: `5`
- `--num_videos`: Number of videos to record during evaluation. Default: `3`

### Experiment Flags
- `--suffix`: Custom suffix to append to the experiment name
- `--logdir`: Base directory for storing logs and checkpoints. Default: `logs`

### Hyperparameter Flags (SAC/TD3)
- `--learning_rate`: Learning rate. Default: `3e-4`
- `--buffer_size`: Replay buffer size. Default: `1000000`
- `--learning_starts`: Steps before learning starts. Default: `10000`
- `--batch_size`: Batch size for training. Default: `256`
- `--tau`: Target network update rate. Default: `0.005`
- `--gamma`: Discount factor. Default: `0.99`

### Checkpoint Flags
- `--checkpoint_freq`: Save checkpoint every N steps. Default: `25000`
- `--eval_freq`: Evaluate policy every N steps during training. Default: `10000`

## Directory Structure

After running training, the logs are organized as follows:

```
logs/
└── {env_name}-{timestamp}-{suffix}/
    ├── config.json                    # Training configuration
    ├── final_model.zip                # Final trained model
    ├── vec_normalize.pkl              # Observation normalization statistics
    ├── checkpoints/                   # Periodic checkpoints during training
    │   ├── model_25000_steps.zip
    │   ├── model_50000_steps.zip
    │   ├── rl_model_25000_steps.zip  # Replay buffer
    │   └── ...
    ├── best_model/                    # Best model based on evaluation
    │   └── best_model.zip
    ├── eval_logs/                     # Evaluation metrics during training
    │   ├── evaluations.npz
    │   └── evaluations.txt
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
- **`best_model/`**: Contains the best-performing model based on evaluation metrics
- **`videos/`**: MP4 recordings of the agent's behavior during evaluation

## Monitoring Training

TensorBoard logs are automatically saved during training. To monitor progress:

```bash
tensorboard --logdir logs/{your_run_name}/tensorboard
```

Then open your browser to `http://localhost:6006` to view training metrics in real-time.

## Troubleshooting

### JAX GPU Recognition Issue
There is a known bug where JAX may not recognize the GPU. A temporary fix:

```bash
sudo rmmod nvidia_uvm
sudo modprobe nvidia_uvm
```

See this [issue discussion](https://github.com/jax-ml/jax/issues/28980) for more details.

### Model Not Loading
Ensure you're using the exact run name (with timestamp) when loading checkpoints. Check the `logs/` directory for available runs.
