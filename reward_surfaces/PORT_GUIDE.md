# Reward Surfaces Port - Complete Guide

## What Was Ported

### Core Files (610 lines total)

**From reward-surfaces → To MPC-Injection/reward_surfaces**

1. **Core Logic** (`reward_surfaces/core/`)
   - `evaluator.py` - Abstract interface (45 lines)
   - `surface_generation.py` - Filter normalization & job generation (110 lines)

2. **Utilities** (`reward_surfaces/utils/`)
   - `io.py` - NPZ file I/O (25 lines)
   - `compute_stats.py` - Evaluation statistics (50 lines)

3. **Plotting** (`reward_surfaces/plotting/`)
   - `plot_plane.py` - Surface visualization (200 lines)

4. **Scripts** (`scripts/`)
   - `generate_plane_jobs.py` - Setup evaluation grid (110 lines)
   - `eval_plane_job.py` - Evaluate single point (70 lines)
   - `job_results_to_csv.py` - Aggregate results (40 lines)
   - `run_jobs_multiproc.py` - Parallel execution (40 lines)
   - `plot_plane.py` - CLI for plotting (40 lines)
   - `sbx_adapter.py` - **NEW** - SBX/JAX integration (180 lines)

### What Was NOT Ported (and why)

❌ **Old SB3 trainer infrastructure** (~2000 lines)
   - `experiment_manager.py` - You use train.py instead
   - `make_agent.py` - You have your own agent creation
   - Rainbow/Atari support - Not relevant

❌ **Old evaluation framework** (~800 lines)
   - `eval_policy_hess.py` - Hessian calculation (overkill)
   - Gradient search - Not needed for basic surfaces

❌ **Legacy gym code**
   - gym 0.19.0 compatibility layers - You use gymnasium

## Directory Structure

```
MPC-Injection/
├── mpc_rl/
│   └── train.py              # Your training script (unchanged)
├── logs/                          # Your training outputs (unchanged)
└── reward_surfaces/               # NEW - Minimal reward surface package
    ├── README.md
    ├── PORT_GUIDE.md
    ├── reward_surfaces/
    │   ├── __init__.py
    │   ├── core/
    │   │   ├── __init__.py
    │   │   ├── evaluator.py      # Abstract interface
    │   │   └── surface_generation.py  # Core algorithm
    │   ├── utils/
    │   │   ├── __init__.py
    │   │   ├── io.py              # File I/O
    │   │   └── compute_stats.py   # Statistics
    │   └── plotting/
    │       ├── __init__.py
    │       └── plot_plane.py      # Visualization
    └── scripts/
        ├── __init__.py
        ├── sbx_adapter.py         # NEW - Connects SBX to surfaces
        ├── generate_plane_jobs.py # Step 1: Setup
        ├── eval_plane_job.py      # Step 2: Evaluate
        ├── job_results_to_csv.py  # Step 3: Aggregate
        ├── run_jobs_multiproc.py  # Parallel runner
        └── plot_plane.py          # Step 4: Visualize
```

## Key Adaptations

### 1. SBX/JAX Integration (`sbx_adapter.py`)

**Challenge**: Original code used PyTorch. Your models use JAX.

**Solution**: Created `SBXRewardSurfaceEvaluator` that:
- Extracts JAX parameters → converts to numpy
- Sets numpy parameters → converts back to JAX
- Handles dm_control environment creation
- Auto-detects algorithm from model path
- Manages VecNormalize loading

### 2. Checkpoint Structure

**Original**: 
```
runs/cartpole_checkpoints/
├── info.json
├── 0010000/
│   ├── checkpoint.zip
│   └── parameters.th
└── best/
    └── checkpoint.zip
```

**Your Structure**:
```
logs/walker-walk-SAC-MPC-TIMESTAMP/
├── config.json                    # Different name, same purpose
├── checkpoints/
│   └── model_100000_steps.zip     # Different naming
├── best_model/
│   └── best_model.zip             # Different path
└── vec_normalize.pkl              # Important for evaluation
```

**Adaptation**: Scripts now:
- Look for `config.json` instead of `info.json`
- Handle `best_model/best_model.zip` path
- Auto-detect `vec_normalize.pkl` location

### 3. Environment Creation

**Original**: Used old gym + experiment manager

**Your Setup**: Uses dm_control + shimmy

**Solution**: `sbx_adapter.py` creates environments your way:
```python
dm_env = suite.load(domain_name=domain, task_name=task)
gym_env = DmControlCompatibilityV0(dm_env)
gym_env = FlattenObservation(gym_env)
```

## Usage Workflow

### Quick Start (11x11 grid, ~2 minutes)

```bash
# 1. Activate your environment and navigate to MPC-Injection
source ~/.bash_conda && conda activate mpc-rl
cd /home/roy/MPC-Injection

# 2. Pick a trained model
MODEL_DIR="logs/SAC-MPC-walker-runs/1st_run/walker-walk-SAC-MPC-20251105-100225-percentage-0pct"

# 3. Generate jobs (11x11 = 121 evaluations)
python reward_surfaces/scripts/generate_plane_jobs.py \
  $MODEL_DIR/best_model/ \
  logs/test_surface/ \
  --grid-size=11 \
  --magnitude=0.5 \
  --num-episodes=10

# 4. Run evaluations (parallel)
python reward_surfaces/scripts/run_jobs_multiproc.py \
  --num-cpus=4 \
  logs/test_surface/jobs.sh

# 5. Convert to CSV
python reward_surfaces/scripts/job_results_to_csv.py \
  logs/test_surface/

# 6. Plot
python reward_surfaces/scripts/plot_plane.py \
  logs/test_surface/results.csv \
  --outname=logs/test_surface/walker_surface \
  --env-name="Walker-Walk SAC-MPC" \
  --type=all
```

### Production Run (31x31 grid, ~1 hour)

```bash
# Same as above but:
--grid-size=31        # 961 evaluations
--magnitude=1.0       # Larger perturbations
--num-episodes=50     # More accurate
--num-cpus=8          # More parallelism
```

## What Each Script Does

### `generate_plane_jobs.py`
**Input**: Trained checkpoint  
**Output**: 
- `jobs.sh` - Shell script with all commands
- `dir1.npz`, `dir2.npz` - Random directions  
- `info.json` - Metadata

**What it does**:
1. Loads your model
2. Generates 2 random filter-normalized directions
3. Creates grid of (offset1, offset2) pairs
4. Writes one command per grid point

### `eval_plane_job.py`
**Input**: Job directory + offset  
**Output**: `results/{offset1},{offset2}.json`

**What it does**:
1. Loads model
2. Perturbs weights: `θ' = θ + α*dir1 + β*dir2`
3. Runs N episodes
4. Saves statistics

### `job_results_to_csv.py`
**Input**: Job directory with results/*.json  
**Output**: `results.csv`

**What it does**:
- Collects all JSON files
- Combines into single DataFrame
- Sorts by coordinates

### `plot_plane.py`
**Input**: `results.csv`  
**Output**: PNG plots

**What it does**:
- Loads CSV
- Creates meshgrid
- Generates 3D surface / heatmap / contour plots

## Parameters Explained

### `--grid-size`
- **Must be odd** (e.g., 11, 21, 31)
- Center point is the original model
- 11x11 = 121 evaluations (~5 mins)
- 31x31 = 961 evaluations (~30-60 mins)

### `--magnitude`
- How far to perturb parameters
- 0.5 = conservative (stays close to trained model)
- 1.0 = standard (original paper default)
- 2.0+ = aggressive (may break agent)

### `--num-episodes`
- Evaluation accuracy vs speed tradeoff
- 10 = fast but noisy
- 50 = good balance (recommended)
- 100+ = very accurate but slow

## Troubleshooting

### "JAX device error" during parallel execution
**Fix**: `eval_plane_job.py` forces CPU mode. If still issues:
```python
# Add to your .bashrc
export JAX_PLATFORMS="cpu"
```

### "No module named 'mpc_rl'"
**Fix**: Run scripts from `/home/roy/MPC-Injection/` directory

### "VecNormalize not found"
**Fix**: Specify manually:
```bash
--vecnormalize=$MODEL_DIR/vec_normalize.pkl
```

### Weird parameter shapes
**Check**: JAX parameter extraction in `sbx_adapter.py::get_weights()`
- Currently handles actor/qf/qf1/qf2
- May need adjustment for PPO (value function structure differs)

## Files You Can Safely Delete

From the original `reward-surfaces` repo, you don't need:
- `old_experiments/` - All of it
- `reward_surfaces/agents/` - All of it (except concepts)
- `reward_surfaces/algorithms/eval_policy_hess.py`
- `reward_surfaces/hyperparams/` - All YAML files
- `test/` - Tests for old code
- `vector/` - Old vectorized envs

## Next Steps

1. **Test on small grid** (11x11) to verify everything works
2. **Check parameter extraction** - Print `get_weights()` output to verify shape
3. **Production run** (31x31) on your best Walker model
4. **Compare**: 0% MPC vs 50% MPC vs 100% MPC surfaces

## Performance Notes

**Timing (approximate)**:
- 11x11 grid, 10 episodes, 4 CPUs: ~5 minutes
- 31x31 grid, 50 episodes, 8 CPUs: ~45-60 minutes

**Disk Space**:
- Each result file: ~1KB
- 961 results: ~1MB
- Direction files: ~500KB each
- Total per surface: ~2-3MB

**Memory**:
- Each eval process: ~500MB-1GB (JAX + model + env)
- Safe to run 8-16 parallel on most machines
