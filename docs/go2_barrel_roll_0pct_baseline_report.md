# Go2 barrel-roll 0% MPC-injection baseline

## Setup and result

I trained one seed-1 SAC-MPC policy for 500,000 environment steps with the G9
schema-v3 barrel-roll recipe and `--percentage=0`. The robot, reward, episode
horizon, network/hyperparameters, four training environments, Go2 sysID,
disabled domain randomization, and 10k-step/100-seed validation protocol match
the successful 25% run. The official successful 25% policy is **seed 1 at
220,000 steps**, not 230,000: seed 2 was the 230,000-step policy. I therefore
used 220,000 for the matched baseline evaluation.

The 25% campaign saved periodic checkpoints every 25k and retained 220k through
its best-model callback. A no-injection run would not necessarily select 220k,
so I changed only the periodic save interval to 20k to guarantee an exact 220k
model/normalization pair. This changes checkpoint I/O, not learning updates.

The run exited zero after 49:27.80, used at most 3,563,452 KiB RSS, and reported
zero swaps. Diagnostics completed with 125,000 finite critic checks and a final
MPC replay percentage of exactly 0.0%; no replay-buffer files were saved. The
220k policy scored 86/100 on validation seeds 2,000,000–2,000,099; the 500k
policy scored 100/100. On the ten video seeds 3,000,000–3,000,009, both policies
scored 10/10. Thus the initial hypothesis was not supported: this RL-only
baseline learned classifier-valid, visually recognizable barrel rolls. It was
extremely unstable during training (for example 100/100 at 290k, 0/100 at 300k,
then 100/100 at 310k). A frame-sequence spot-check of seed 3,000,000 from each
video set showed a lateral roll and upright return, consistent with the
classifier.

## Commands run

Training (stdout and `/usr/bin/time -v` were retained):

```bash
mkdir -p logs/go2_barrel_roll_0pct_baseline
/usr/bin/time -v -o logs/go2_barrel_roll_0pct_baseline/training_time.txt \
  conda run --no-capture-output -n mpc-rl python mpc_rl/train.py \
  --env_name=quadruped-barrel_roll --robot=go2 --algorithm=SAC-MPC \
  --total_timesteps=500000 --num_envs=4 --max_episode_steps=125 \
  --learning_rate=0.0003 --buffer_size=1000000 --learning_starts=10000 \
  --batch_size=256 --tau=0.005 --gamma=0.99 --gradient_steps=-1 \
  --inject_type=percentage --percentage=0 --random_select=True \
  --quadruped_mpc_replay_mode=direct --data_dir=data/go2_barrel_roll/v3 \
  --domain_rand=False --domain_rand_config_type=disabled --use_go2_sysid=True \
  --checkpoint_freq=20000 --eval_freq=10000 \
  --save_replay_buffer_checkpoints=False --save_replay_buffer_final=False \
  --enable_logging=True --logdir=logs/go2_barrel_roll_0pct_baseline \
  --seed=1 --suffix=go2-barrel-roll-v3-baseline \
  > logs/go2_barrel_roll_0pct_baseline/training_stdout.log 2>&1
```

Ten videos per policy and their evaluation/hash summary:

```bash
conda run --no-capture-output -n mpc-rl python \
  mpc_rl/evaluate_go2_barrel_roll_0pct.py \
  --run-dir logs/go2_barrel_roll_0pct_baseline/quadruped-barrel_roll-SAC-MPC-20260808-103700-percentage-0pct-go2-barrel-roll-v3-baseline \
  --checkpoint-step 220000
```

One deterministic trajectory from each policy (both use reset seed 3,000,000):

```bash
conda run --no-capture-output -n mpc-rl python \
  body_trajs/record_go2_barrel_roll_policy.py \
  --run-dir logs/go2_barrel_roll_0pct_baseline --checkpoint-step 220000 \
  --seed 3000000 --output-dir logs/go2_barrel_roll_0pct_baseline/trajectories

conda run --no-capture-output -n mpc-rl python \
  body_trajs/record_go2_barrel_roll_policy.py \
  --run-dir logs/go2_barrel_roll_0pct_baseline --final-model \
  --seed 3000000 --output-dir logs/go2_barrel_roll_0pct_baseline/trajectories
```

## Artifact locations

The concrete run directory is
`logs/go2_barrel_roll_0pct_baseline/quadruped-barrel_roll-SAC-MPC-20260808-103700-percentage-0pct-go2-barrel-roll-v3-baseline/`.

- Models: `checkpoints/model_220000_steps.zip`, its matching
  `model_vecnormalize_220000_steps.pkl`, `final_model.zip`, and
  `vec_normalize.pkl`.
- Training evidence: `config.json`, `barrel_roll_eval_history.jsonl`,
  `barrel_roll_pilot_diagnostics.json`, and `tensorboard/`. Process output and
  resources are one level above in `training_stdout.log` and
  `training_time.txt`.
- Videos: `baseline_evaluation/checkpoint_220000_videos/` and
  `baseline_evaluation/final_model_videos/`. `baseline_evaluation/summary.json`
  contains evaluation outcomes and SHA-256 hashes for both model pairs and all
  20 videos. Every video is 640x480, 50 fps, and 125 frames.
- Trajectories: under `logs/go2_barrel_roll_0pct_baseline/trajectories/.../` as
  `training_seed1_checkpoint_step220000_rollout_seed3000000.npz` and
  `training_seed1_final_model_final_rollout_seed3000000.npz`. Each contains 125
  actions and 126 state frames and embeds its source-model hashes.

## View both trajectories from a laptop with Viser

From a laptop terminal, create the SSH tunnel and log in:

```bash
ssh -L 8081:localhost:8081 roy@10.10.20.3
```

In that remote shell, run:

```bash
cd /home/roy/MPC-Injection
TRAJ_DIR=logs/go2_barrel_roll_0pct_baseline/trajectories/quadruped-barrel_roll-SAC-MPC-20260808-103700-percentage-0pct-go2-barrel-roll-v3-baseline
conda run --no-capture-output -n mpc-rl python \
  viser/viser_quadruped_viz_trajs.py \
  --trajectory1 "$TRAJ_DIR/training_seed1_checkpoint_step220000_rollout_seed3000000.npz" \
  --trajectory2 "$TRAJ_DIR/training_seed1_final_model_final_rollout_seed3000000.npz" \
  --port 8081
```

Then open `http://localhost:8081` in the laptop browser. Keep the SSH session
and Viser process running; press Ctrl+C when finished. If laptop port 8081 is
occupied, use `ssh -L 18081:localhost:8081 ...` and browse to
`http://localhost:18081` instead. This tunnel/viewer path was smoke-tested with
the two saved baseline trajectories.
