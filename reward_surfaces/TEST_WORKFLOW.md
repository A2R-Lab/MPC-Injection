# Reward Surface Evaluation - Complete Test Workflow

This document shows the complete working workflow tested on December 16, 2024.

## Prerequisites

```bash
cd /path/to/MPC-Injection
conda activate "$PWD/.conda/mpc-injection"
```

## Step 1: Generate Evaluation Jobs

```bash
python reward_surfaces/scripts/generate_plane_jobs.py \
    logs/SAC-MPC-walker-runs/1st_run/walker-walk-SAC-MPC-20251105-100225-percentage-0pct/best_model \
    demo/test_surface \
    --grid-size 5
```

**What it does:**
- Loads your trained SAC-MPC model
- Generates 2 random filter-normalized perturbation directions
- Creates a 5x5 grid of evaluation jobs (25 total)
- Saves directions as `dir1.npz` and `dir2.npz`
- Creates `jobs.sh` with all evaluation commands

**Output files:**
- `demo/test_surface/dir1.npz` - First perturbation direction (1.8 MB)
- `demo/test_surface/dir2.npz` - Second perturbation direction (1.8 MB)
- `demo/test_surface/info.json` - Metadata (grid size, model path, etc.)
- `demo/test_surface/jobs.sh` - Shell script with 25 evaluation commands
- `demo/test_surface/results/` - Empty directory for results

## Step 2: Run Evaluation Jobs

### Option A: Run Single Job (for testing)

```bash
python reward_surfaces/scripts/eval_plane_job.py demo/test_surface --offset1=0 --offset2=0
```

**Expected output:**
```
Loading model from logs/.../best_model.zip
Loaded SAC-MPC model...
Evaluating at offset (0.0, 0.0) for 50 episodes...
Results saved to demo/test_surface/results/0.0,0.0.json
Mean reward: 977.23 ± 10.01
```

### Option B: Run All Jobs Sequentially

```bash
bash demo/test_surface/jobs.sh
```

**Time estimate:** ~20-30 seconds per job × 25 jobs = ~10-15 minutes

### Option C: Run Jobs in Parallel (Recommended)

```bash
python reward_surfaces/scripts/run_jobs_multiproc.py --num-cpus=8 demo/test_surface/jobs.sh
```

**Time estimate:** ~2-3 minutes with 8 CPUs

## Step 3: Aggregate Results to CSV

```bash
python reward_surfaces/scripts/job_results_to_csv.py demo/test_surface
```

**Expected output:**
```
Found 25 result files
Saved 25 results to demo/test_surface/results.csv

Available columns: episode_rewards, episode_std_rewards, ...
Reward statistics:
  Mean: 419.73
  Std:  324.32
  Min:  185.55
  Max:  977.23
```

**Output file:**
- `demo/test_surface/results.csv` - CSV with columns: dim0, dim1, offset1, offset2, episode_rewards, etc.

## Step 4: Generate Reward Surface Plot

```bash
python reward_surfaces/scripts/plot_plane.py \
    demo/test_surface/results.csv \
    --outname demo/test_surface/reward_surface.png
```

**Expected output:**
```
Plotting reward surface from demo/test_surface/results.csv
Saved: demo/test_surface/reward_surface.png_3dsurface.png
```

**Output file:**
- `demo/test_surface/reward_surface.png_3dsurface.png` - 3D surface plot (~600 KB)

## Verified Test Results (Dec 16, 2024)

Tested with:
- Model: `walker-walk-SAC-MPC-20251105-100225-percentage-0pct`
- Grid size: 5×5 (25 evaluation points)
- Episodes per point: 50

Sample results:
- Center (0, 0): **977.23 ± 10.01** (best - original trained model)
- Corner (-2, -2): 309.22 ± 60.04
- Corner (2, 2): 413.89 ± 125.58
- Corner (-2, 2): 185.55 ± 12.94 (worst)
- Corner (2, -2): 212.76 ± 20.74

**Total time:** ~30 seconds (5 jobs tested manually)

## Common Options

### Grid Size
- `--grid-size 5` - Quick test (25 jobs)
- `--grid-size 11` - Medium detail (121 jobs) 
- `--grid-size 31` - High detail (961 jobs, ~8 hours)

### Number of Episodes
- `--num-episodes 10` - Fast preview
- `--num-episodes 50` - Default, good balance
- `--num-episodes 100` - More accurate statistics

### Plot Types
```bash
python reward_surfaces/scripts/plot_plane.py demo/test_surface/results.csv \
    --outname output.png \
    --type all  # Creates mesh, heat, contour, and contourf plots
```

## Troubleshooting

**ModuleNotFoundError: No module named 'mpc_rl'**
- Solution: Run from `/home/roy/MPC-Injection` directory

**AttributeError: actor.params**
- Fixed! Now uses `actor_state.params` and `qf_state.params`

**ValueError: Only 1, 2, 4 dimensional tensors supported**
- Fixed! Now supports 3D tensors for JAX/Flax models

**Shape mismatch when applying perturbations**
- Fixed! Uses `readz()` function to maintain parameter order
