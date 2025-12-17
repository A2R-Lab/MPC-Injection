# Reward Surfaces for MPC-RL

Visualize reward surfaces for your trained SBX (Stable-Baselines3 + JAX) agents.

## Prerequisites

**Python packages needed** (should already be in mpc-rl environment):
- numpy, pandas, matplotlib, seaborn, tqdm

If missing, install with:
```bash
pip install pandas matplotlib seaborn tqdm
```

**Usage**:
```bash
source ~/.bash_conda && conda activate mpc-rl
cd /home/roy/MPC-RL
```

**Note**: No package installation required - scripts work directly via `sys.path` manipulation.

## Usage

### 1. Train Your Agent

First, train your agent as usual with `train_sbx.py`:

```bash
python mpc_rl/train_sbx.py \
  --env_name=walker-walk \
  --algorithm=SAC-MPC \
  --total_timesteps=500000 \
  --checkpoint_freq=25000
```

This creates checkpoints in `logs/walker-walk-SAC-MPC-TIMESTAMP/`

### 2. Generate Reward Surface Jobs

Generate evaluation jobs for a 31x31 grid around your best model:

```bash
python reward_surfaces/scripts/generate_plane_jobs.py \
  logs/walker-walk-SAC-MPC-TIMESTAMP/best_model/ \
  logs/walker_surface/ \
  --grid-size=31 \
  --magnitude=1.0 \
  --num-episodes=50
```

This creates:
- `logs/walker_surface/jobs.sh` - Shell script with all evaluation commands
- `logs/walker_surface/dir1.npz` - First random direction
- `logs/walker_surface/dir2.npz` - Second random direction
- `logs/walker_surface/info.json` - Metadata

### 3. Run Evaluations

Run all evaluations in parallel:

```bash
python reward_surfaces/scripts/run_jobs_multiproc.py \
  --num-cpus=8 \
  logs/walker_surface/jobs.sh
```

Or run sequentially:

```bash
bash logs/walker_surface/jobs.sh
```

### 4. Convert Results to CSV

Aggregate individual results:

```bash
python reward_surfaces/scripts/job_results_to_csv.py \
  logs/walker_surface/
```

This creates `logs/walker_surface/results.csv`

### 5. Plot the Surface

Generate visualizations:

```bash
python reward_surfaces/scripts/plot_plane.py \
  logs/walker_surface/results.csv \
  --outname=walker_surface \
  --env-name="Walker-Walk" \
  --type=mesh
```

Options for `--type`:
- `mesh` - 3D surface plot (default)
- `heat` - 2D heatmap
- `contour` - 2D contour lines
- `contourf` - Filled 2D contours
- `all` - Generate all plot types

Options for `--key`:
- `episode_rewards` - Mean episode rewards (default)
- `episode_std_rewards` - Standard deviation
- `episode_stderr_rewards` - Standard error

## Complete Example

```bash
# Activate environment and navigate to MPC-RL directory
source ~/.bash_conda && conda activate mpc-rl
cd /home/roy/MPC-RL

# Generate surface
python reward_surfaces/scripts/generate_plane_jobs.py \
  logs/SAC-MPC-walker-stand_only_reward/1st_run/walker-walk-SAC-MPC-20251208-145319-percentage-0pct/best_model/ \
  plots/sac_mpc_walker_stand_only_reward_0_reward_surface/ \
  --grid-size=31 \
  --magnitude=0.5 \
  --num-episodes=25

# Run evaluations (faster on small grid for testing)
python reward_surfaces/scripts/run_jobs_multiproc.py \
  --num-cpus=4 \
  logs/sac_mpc_walker_stand_only_reward_0_reward_surface/jobs.sh

# Convert to CSV
python reward_surfaces/scripts/job_results_to_csv.py \
  plots/sac_mpc_walker_stand_only_reward_0_reward_surface/

# Plot
python reward_surfaces/scripts/plot_plane.py \
  plots/sac_mpc_walker_stand_only_reward_0_reward_surface/results.csv \
  --outname=plots/sac_mpc_walker_stand_only_reward_0_reward_surface/surface \
  --env-name="SAC-MPC Walker Walk 0pct" \
  --type=all
```

## Notes

- **Grid Size**: Use odd numbers (e.g., 11, 21, 31). Larger grids are more detailed but take longer.
- **Magnitude**: Controls how far to perturb parameters. Start with 0.5-1.0.
- **Num Episodes**: More episodes = more accurate but slower. 25-50 is usually sufficient.
- **VecNormalize**: Automatically detected from checkpoint directory.

## Troubleshooting

**JAX/GPU Issues**: The eval_plane_job script forces CPU mode to avoid GPU memory conflicts in parallel execution.

**Missing VecNormalize**: The script looks for `vec_normalize.pkl` in the checkpoint's parent directory. Specify manually with `--vecnormalize` if needed.

**Import Errors**: Make sure you're in the MPC-RL directory and the mpc-rl conda environment is activated.

# Reward Surfaces Code Explained

## Overview

This code visualizes the **reward landscape** around a trained RL policy by perturbing its parameters in random directions and measuring performance. This code is ported from ["Cliff Diving: Exploring Reward Surfaces in RL Environments"](https://arxiv.org/abs/2205.07015), which implements the filter normalization technique from ["Visualizing the Loss Landscape of Neural Nets" (Li et al. 2018)](https://arxiv.org/abs/1712.09913), adapted for reward-based RL evaluation instead of loss-based supervised learning.

## Key Concept: Filter Normalization

The core idea is to generate random parameter perturbations that respect the scale of each network layer:

1. **Problem**: Naive random directions treat all parameters equally, but layer weights have vastly different magnitudes
2. **Solution**: For each parameter tensor (weight matrix), generate a random direction with the same "filter-wise" norm structure
3. **Result**: Perturbations are proportional to the existing parameter magnitudes, making the visualization meaningful

For a fully connected layer with weights `W` of shape `(input_dim, output_dim)`:
- Generate random matrix `D` of same shape
- Normalize each **output filter** (column): `D[:, i] /= ||D[:, i]||`
- Scale to match original: `D[:, i] *= ||W[:, i]||`

This ensures each output neuron's weights are perturbed proportionally to their trained magnitude.

## Architecture

```bash
MPC-RL/reward_surfaces/
├── README.md                         # Usage guide  
├── PORT_GUIDE.md                     # Technical details
├── reward_surfaces/                  # Core package
│   ├── __init__.py
│   ├── core/
│   │   ├── evaluator.py             # Abstract interface
│   │   └── surface_generation.py    # Filter normalization algorithm
│   ├── utils/
│   │   ├── io.py                    # NPZ file handling
│   │   └── compute_stats.py         # Episode statistics
│   └── plotting/
│       └── plot_plane.py            # 3D surface/heatmap visualization
└── scripts/                         # Executables (command-line tools)
    ├── sbx_adapter.py               # JAX/SBX model interface
    ├── generate_plane_jobs.py       # Step 1: Setup evaluation grid
    ├── eval_plane_job.py            # Step 2: Evaluate single point
    ├── job_results_to_csv.py        # Step 3: Aggregate results
    ├── run_jobs_multiproc.py        # Parallel job runner
    └── plot_plane.py                # Step 4: Generate plots
```

## Code Flow

### Step 1: Generate Evaluation Grid (`generate_plane_jobs.py`)

**Input**: Trained model checkpoint  
**Output**: Direction vectors and evaluation job list

```python
# Load model
evaluator = SBXRewardSurfaceEvaluator(model_path, domain, task)

# Extract current parameters
weights = evaluator.get_weights()  # List of numpy arrays

# Generate 2 random orthogonal directions via filter normalization
dir1_vec = [filter_normalize(w) for w in weights]
dir2_vec = [filter_normalize(w) for w in weights]

# Save directions
np.savez("dir1.npz", *dir1_vec)
np.savez("dir2.npz", *dir2_vec)

# Create grid: for 5x5 grid, evaluate at offsets (-2,-2) to (2,2)
for x in range(-grid_size//2, grid_size//2 + 1):
    for y in range(-grid_size//2, grid_size//2 + 1):
        jobs.append(f"eval_plane_job.py --offset1={x} --offset2={y}")
```

**Key file: `surface_generation.py`**

The `filter_normalize()` function handles different parameter shapes:

```python
def filter_normalize(param: np.ndarray) -> np.ndarray:
    if ndim == 1:  # Biases
        return np.zeros_like(param)  # Don't perturb biases
    
    elif ndim == 2:  # Fully connected: (input_dim, output_dim)
        direction = np.random.normal(size=param.shape)
        # Normalize along input dimension (axis=0)
        direction /= np.sqrt(np.sum(direction**2, axis=0, keepdims=True))
        # Scale to match original parameter norms
        direction *= np.sqrt(np.sum(param**2, axis=0, keepdims=True))
        return direction
    
    elif ndim == 3:  # JAX ensemble: (n_ensemble, input_dim, output_dim)
        # E.g., SAC uses 2 Q-networks stored as (2, 256, 256)
        direction = np.random.normal(size=param.shape)
        # Normalize along input dimension (axis=1) for each ensemble member
        direction /= np.sqrt(np.sum(direction**2, axis=1, keepdims=True))
        direction *= np.sqrt(np.sum(param**2, axis=1, keepdims=True))
        return direction
```

### Step 2: Evaluate Single Point (`eval_plane_job.py`)

**Input**: Job directory with `dir1.npz`, `dir2.npz`, and offsets  
**Output**: JSON file with episode rewards and statistics

```python
# Load base model and directions
base_weights = evaluator.get_weights()
dir1_vec = readz("dir1.npz")  # Maintains order
dir2_vec = readz("dir2.npz")

# Apply perturbation: θ' = θ + α·d1 + β·d2
alpha = offset1 / (grid_size // 2)  # Normalize to [-1, 1]
beta = offset2 / (grid_size // 2)

perturbed_weights = []
for base, d1, d2 in zip(base_weights, dir1_vec, dir2_vec):
    perturbed = base + alpha * d1 + beta * d2
    perturbed_weights.append(perturbed)

# Set perturbed parameters
evaluator.set_weights(perturbed_weights)

# Run episodes and collect rewards
results = evaluator.evaluate(num_episodes=50)

# Save: {"episode_rewards": 850.3, "episode_std_rewards": 45.2, ...}
save_json(results, f"results/{alpha},{beta}.json")
```

**Key file: `sbx_adapter.py`**

This bridges SBX/JAX models with the reward surface framework:

```python
class SBXRewardSurfaceEvaluator(RewardSurfaceEvaluator):
    def get_weights(self) -> List[np.ndarray]:
        """Extract JAX parameters as numpy arrays."""
        # SBX stores params in Flax TrainState objects
        actor_params = self.model.policy.actor_state.params
        critic_params = self.model.policy.qf_state.params
        
        # Flatten JAX pytree to list
        actor_flat = jax.tree_util.tree_leaves(actor_params)
        critic_flat = jax.tree_util.tree_leaves(critic_params)
        
        return [np.array(p) for p in actor_flat + critic_flat]
    
    def set_weights(self, weights: List[np.ndarray]):
        """Restore parameters to JAX model."""
        # Reconstruct pytree structure and update TrainState
        new_actor_params = reconstruct_tree(weights[:n_actor])
        new_qf_params = reconstruct_tree(weights[n_actor:])
        
        self.model.policy.actor_state = \
            self.model.policy.actor_state.replace(params=new_actor_params)
        self.model.policy.qf_state = \
            self.model.policy.qf_state.replace(params=new_qf_params)
    
    def evaluate(self, num_episodes: int) -> Dict[str, float]:
        """Run episodes and compute statistics."""
        episode_rewards = []
        for _ in range(num_episodes):
            obs = self.env.reset()
            done = False
            total_reward = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _ = self.env.step(action)
                total_reward += reward[0]
            
            episode_rewards.append(total_reward)
        
        return {
            "episode_rewards": np.mean(episode_rewards),
            "episode_std_rewards": np.std(episode_rewards),
            ...
        }
```

### Step 3: Aggregate Results (`job_results_to_csv.py`)

**Input**: Directory with `results/*.json` files  
**Output**: Single CSV with all evaluation points

```python
# Load all result files
results = []
for json_file in Path("results").glob("*.json"):
    data = json.load(json_file)
    results.append({
        "dim0": data["dim0"],           # Normalized offset along dir1
        "dim1": data["dim1"],           # Normalized offset along dir2
        "offset1": data["offset1"],     # Grid offset along dir1
        "offset2": data["offset2"],     # Grid offset along dir2
        "episode_rewards": data["episode_rewards"],
        "episode_std_rewards": data["episode_std_rewards"],
        ...
    })

# Save as CSV
df = pd.DataFrame(results)
df.to_csv("results.csv", index=False)
```

Result format:
```
dim0,dim1,offset1,offset2,episode_rewards,episode_std_rewards,...
-1.0,-1.0,-2,-2,309.22,60.04,...
-1.0,-0.5,-2,-1,425.11,35.22,...
...
0.0,0.0,0,0,977.23,10.01,...    <- Original model (best performance)
...
```

### Step 4: Visualize (`plot_plane.py`)

**Input**: CSV with grid results  
**Output**: 3D surface plot, heatmap, or contours

```python
# Load data
df = pd.read_csv("results.csv")

# Pivot to 2D grid
grid = df.pivot(index="dim0", columns="dim1", values="episode_rewards")

# Create 3D surface plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(grid.columns, grid.index)
ax.plot_surface(X, Y, grid.values, cmap='viridis')
ax.set_xlabel("Direction 1")
ax.set_ylabel("Direction 2")
ax.set_zlabel("Mean Episode Reward")
plt.savefig("reward_surface_3d.png")
```