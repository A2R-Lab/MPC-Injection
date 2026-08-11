# Go2 barrel-roll G9 artifacts and reproducibility

Run all remote commands from the repository root:

```bash
cd /home/roy/MPC-RL
```

The completed result is the 25%-MPC-injection G9 campaign. A separate 0%-injection baseline is currently running under `logs/go2_barrel_roll_0pct_baseline`; do not launch another compute-heavy job until it finishes.

## Policies, evaluations, and videos

The selected policy is `best_model/best_model.zip`, not `final_model.zip`. Its matching `vec_normalize.pkl` is required.

| Training seed | Selected step | Run directory |
| ---: | ---: | --- |
| 1 (official) | 220,000 | `logs/go2_barrel_roll_g9_production/runs/seed1/quadruped-barrel_roll-SAC-MPC-20260807-224837-percentage-25pct-go2-barrel-roll-v3` |
| 2 | 230,000 | `logs/go2_barrel_roll_g9_production/runs/seed2/quadruped-barrel_roll-SAC-MPC-20260807-224837-percentage-25pct-go2-barrel-roll-v3` |

The main evidence files are:

- `logs/go2_barrel_roll_g9_production/final_evaluation/summary.json`: compact final result and official artifact hashes.
- `logs/go2_barrel_roll_g9_production/final_evaluation/seed1_final_test.json` and `seed2_final_test.json`: all 100 untouched episodes per policy.
- `logs/go2_barrel_roll_g9_production/final_evaluation/selection_lock.json`: selected-policy and evaluation-protocol lock.
- Each run's `barrel_roll_eval_history.jsonl`: 51 validation evaluations from step 0 to 500,000.
- Each run's `best_model/evaluation.json`: selected checkpoint's 100-seed validation details.
- The campaign root's `campaign_validation.json`, `campaign_provenance.txt`, `seed*_stdout.log`, `seed*_time.txt`, and `resource_samples.jsonl`: configuration, provenance, console, timing, and resource evidence.
- Each run's `barrel_roll_pilot_diagnostics.json`: finite observation, reward, loss, critic-value, progress, and replay-percentage ranges.
- `data/go2_barrel_roll/v3/dataset_summary.json`, manifests, and `checksums.sha256`: MPC dataset statistics and integrity.

Print the final headline result:

```bash
jq '{official_training_seed, training_seed_results, official_artifact}' \
  logs/go2_barrel_roll_g9_production/final_evaluation/summary.json
```

The 20 successful 2.50 s H.264 videos (640x480, 50 fps) are in:

```text
logs/go2_barrel_roll_g9_production/final_evaluation/official_videos/  # seed-1 policy
logs/go2_barrel_roll_g9_production/final_evaluation/seed2_videos/     # seed-2 policy
```

The seed-2 directory and its `summary.json` were added after the frozen official evaluation; all ten use rollout seeds 3,000,000--3,000,009 and are classified successful. The official seed-1 videos and hashes remain recorded in the main `summary.json`.

## Training

The exact completed two-seed campaign was launched with:

```bash
./run_go2_barrel_roll_g9_campaign.sh production
```

Do not run that command against the existing campaign: the runner intentionally refuses to overwrite `logs/go2_barrel_roll_g9_production`. For a single-seed reproduction in a new directory, use the following after the 0% baseline completes (change `--seed` and `--logdir` for another run):

```bash
conda run --no-capture-output -n mpc-rl python mpc_rl/train.py \
  --env_name=quadruped-barrel_roll \
  --robot=go2 \
  --algorithm=SAC-MPC \
  --total_timesteps=500000 \
  --num_envs=4 \
  --max_episode_steps=125 \
  --learning_rate=0.0003 \
  --buffer_size=1000000 \
  --learning_starts=10000 \
  --batch_size=256 \
  --tau=0.005 \
  --gamma=0.99 \
  --gradient_steps=-1 \
  --inject_type=percentage \
  --percentage=25 \
  --random_select=True \
  --quadruped_mpc_replay_mode=direct \
  --data_dir=data/go2_barrel_roll/v3 \
  --domain_rand=False \
  --domain_rand_config_type=disabled \
  --use_go2_sysid=True \
  --checkpoint_freq=25000 \
  --eval_freq=10000 \
  --save_replay_buffer_checkpoints=False \
  --save_replay_buffer_final=False \
  --enable_logging=True \
  --logdir=logs/go2_barrel_roll_g9_reproduction_seed1 \
  --seed=1 \
  --suffix=go2-barrel-roll-v3-reproduction
```

Training automatically performs fixed 100-seed validation at step zero and every 10,000 steps, saves the earliest best-success checkpoint, and evaluates that selected checkpoint after training. Checkpoints are written every 25,000 steps. Replay buffers are intentionally not saved.

## Evaluation

The completed untouched final evaluation is write-once. Inspect its JSON files; do not rerun it or delete `final_evaluation`. For a new, complete two-seed campaign with the same directory topology and no existing `final_evaluation`, the formal evaluator is:

```bash
conda run --no-capture-output -n mpc-rl python \
  mpc_rl/evaluate_go2_barrel_roll_g9.py \
  --campaign-dir=logs/NEW_COMPLETE_G9_CAMPAIGN
```

It validates both selected checkpoints, locks their hashes, scores each once on seeds 3,000,000--3,000,099, and renders ten videos for the official policy. `train.py --play_only` loads `final_model.zip`; it does not evaluate the success-selected `best_model`, so it should not be used to reproduce the reported selected-policy result.

## TensorBoard over SSH

On the remote workstation, start TensorBoard over loopback:

```bash
cd /home/roy/MPC-RL
conda run --no-capture-output -n mpc-rl tensorboard \
  --logdir=logs/go2_barrel_roll_g9_production/runs \
  --host=127.0.0.1 \
  --port=6006
```

On the laptop, open a second terminal and create a tunnel:

```bash
ssh -N -L 6006:127.0.0.1:6006 roy@10.10.20.3
```

Then open `http://127.0.0.1:6006` locally. The event files are under each run's `tensorboard/SAC_1/`. Important series include `eval/barrel_roll_success_rate`, `eval/mean_reward`, `replay_buffer/mpc_percentage_actual`, `reward/{roll_tracking,rate_tracking,action_change,terminal_outcome}`, `barrel_roll/{progress_mean,error_mean,success_fraction,action_clip_fraction,torque_saturation_fraction}`, `eval/failure_reason/*`, `train/{actor_loss,critic_loss,ent_coef}`, and `diagnostics/{q_value_min,q_value_max}`. Validation success is the checkpoint-selection metric; training reward is not.

## Record a policy trajectory for Viser

The recorder uses the normal environment step, including action clipping, the 5 Hz target filter, PD control, and the task termination logic. Record a known-success seed-1 rollout:

```bash
conda run --no-capture-output -n mpc-rl python \
  body_trajs/record_go2_barrel_roll_policy.py \
  --run-dir=logs/go2_barrel_roll_g9_production/runs/seed1 \
  --seed=3000000 \
  --output-dir=logs/go2_barrel_roll_g9_production/viser_trajectories \
  --require-success
```

Record the same rollout from seed 2 by changing `runs/seed1` to `runs/seed2`. To record all ten known-success rollout seeds for seed 1, use:

```bash
for rollout_seed in $(seq 3000000 3000009); do
  conda run --no-capture-output -n mpc-rl python \
    body_trajs/record_go2_barrel_roll_policy.py \
    --run-dir=logs/go2_barrel_roll_g9_production/runs/seed1 \
    --seed="${rollout_seed}" \
    --output-dir=logs/go2_barrel_roll_g9_production/viser_trajectories \
    --require-success
done
```

The first command writes:

```text
logs/go2_barrel_roll_g9_production/viser_trajectories/quadruped-barrel_roll-SAC-MPC-20260807-224837-percentage-25pct-go2-barrel-roll-v3/training_seed1_best_model_step220000_rollout_seed3000000.npz
```

The recorder refuses to overwrite an existing trajectory. Use a different output directory or remove only the specific file after confirming it is disposable.

## View Viser from the laptop

On the remote workstation, start the server:

```bash
cd /home/roy/MPC-RL
conda run --no-capture-output -n mpc-rl python \
  viser/viser_quadruped_viz_trajs.py \
  --trajectory=logs/go2_barrel_roll_g9_production/viser_trajectories/quadruped-barrel_roll-SAC-MPC-20260807-224837-percentage-25pct-go2-barrel-roll-v3/training_seed1_best_model_step220000_rollout_seed3000000.npz \
  --port=8081
```

On the laptop, open a second terminal:

```bash
ssh -N -L 8081:127.0.0.1:8081 roy@10.10.20.3
```

Open `http://127.0.0.1:8081` locally. Keep both the remote Viser process and the laptop tunnel running. To compare two trajectories, replace `--trajectory=...` with `--trajectory1=PATH1 --trajectory2=PATH2`; the viewer supports synchronized dual-robot playback.
