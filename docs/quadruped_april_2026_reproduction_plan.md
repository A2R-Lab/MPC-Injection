# Quadruped April 2026 Reproduction Plan

## Purpose

This document turns the diagnosis in
`docs/quadruped_april_2026_reproduction_diagnosis.md` into an implementation
plan. It is written so another agent can decide whether to reproduce the April
4-5, 2026 quadruped SAC-MPC runs by using the historical code or by adding a
legacy compatibility mode to the current branch.

The core question is whether changing only `run_quadruped_experiments.sh` is
enough.

Answer:

- For exact historical reproduction: yes, use the April commit and only adjust
  the script/log directory. Do not edit `mpc_rl/envs/velocity_tracking_env.py`.
- For current-branch reproduction: no, script-only changes are insufficient.
  Current environment, reward, sysID, and MPC injection behavior differ from the
  April code path, so `velocity_tracking_env.py` and related training/callback
  files need explicit legacy switches.

## Context another agent should know

Original target artifacts:

- Logs: `logs/SAC-MPC-quadruped_x_vel_only_runs_min_rwrd/`
- Run dates: April 4-5, 2026.
- Likely source commit: `25e1a114` from April 3, 2026.
- Algorithm: `SAC-MPC`
- Env: `quadruped-velocity_tracking`
- Timesteps: `1000000`
- Vector envs: `4`
- Seeds: `100 150 200`
- Percentages: `0 25 50`
- Data directory: `data/quadruped/`
- Domain randomization: disabled
- Checkpoint evals: `300000,400000,500000,600000`

Why current runs differ:

- `use_go2_sysid` now exists and defaults true; April had no sysID patch.
- `velocity_tracking_env.py` now low-pass filters action targets before PD;
  April directly used `default_joint_pos + action_scale * action`.
- The SAC-MPC simple reward changed after April 5.
- Quadruped MPC replay insertion changed after April 5 from tiled transitions
  to unique transition batches.
- Current code can inject direct saved transitions when the MPC data contains
  them; April used legacy torque replay from `data/quadruped/`.
- The current `run_quadruped_experiments.sh` defines
  `QUADRUPED_MPC_REPLAY_MODE`, but does not pass it to `mpc_rl/train.py`.

## Track A: exact historical reproduction

Use this track if the goal is to recreate the April results as closely as
possible. This is the recommended path because it avoids approximating historical
behavior through new compatibility flags.

Implementation:

1. Create a separate worktree at `25e1a114`.
2. In that worktree, edit only `run_quadruped_experiments.sh`.
3. Set the experiment parameters to the April artifact values:

   ```bash
   ENV_NAME="quadruped-velocity_tracking"
   ALGORITHM="SAC-MPC"
   TOTAL_TIMESTEPS=1000000
   NUM_ENVS=4
   LEARNING_STARTS=50000
   SAVE_REPLAY_BUFFER_CHECKPOINTS="False"
   SAVE_REPLAY_BUFFER_FINAL="True"
   DOMAIN_RAND="False"
   DOMAIN_RAND_OBS_NOISE=0.0
   DATA_DIR="data/quadruped/"
   BUFFER_SIZE=5000000
   LEARNING_RATE=3e-4
   POLICY_DELAY=2
   BATCH_SIZE=256
   LOG_DIR="logs/SAC-MPC-quadruped_x_vel_only_runs_min_rwrd_repro_25e1a114/"
   PERCENTAGES=(0 25 50)
   SEEDS=(100 150 200)
   CHECKPOINT_EVALS=(300000 400000 500000 600000)
   ```

4. Do not add any of these current-branch flags:

   ```bash
   --use_go2_sysid
   --quadruped_mpc_replay_mode
   --quadruped_action_lpf_cutoff_hz
   --quadruped_simple_reward_mode
   --quadruped_mpc_replay_layout
   ```

5. Keep `data/quadruped/` unchanged. These files were generated April 1-2 and
   match the era of the original runs better than later sysID/direct-transition
   data.

Expected behavior:

- No changes are required in `mpc_rl/envs/velocity_tracking_env.py`.
- The April env already has no action LPF, no Go2 sysID hook, and the old simple
  reward formula.
- The April callback already uses the old tiled replay-buffer insertion path.
- Results should be comparable to the April tensorboard curves, allowing for
  nondeterminism from libraries, hardware, MuJoCo, and PyTorch.

Validation:

- Confirm each new `config.json` matches the April configs except for
  `tensorboard_log` and output directory.
- Compare `rollout/ep_rew_mean`, `eval/mean_reward`, and
  `replay_buffer/mpc_percentage_actual` against the original April runs.
- Re-run `utils/plot_train_pct_comparisons_quadruped.py` against the new log dir
  after training finishes.

## Track B: current-branch legacy compatibility mode

Use this track only if April-like behavior must be runnable from the current
branch. This is more invasive and should preserve current behavior by default.

### Required interface changes

Add these flags to `mpc_rl/train.py` and serialize them into `config.json`:

```text
--quadruped_action_lpf_cutoff_hz
    Float or nullable value. Default: 5.0. April reproduction value: 0 or None.

--quadruped_simple_reward_mode
    Enum: current, april_2026. Default: current.

--quadruped_mpc_replay_layout
    Enum: unique, legacy_tiled. Default: unique.
```

Existing flags to use for April compatibility:

```text
--use_go2_sysid=False
--quadruped_mpc_replay_mode=torque_saved_pd
```

`torque_saved_pd` is the preferred current-branch replay mode for April
compatibility because the old `data/quadruped/` files were generated before
current sysID/current-PD assumptions. `direct` should not be used for April
compatibility when newer direct-transition files are in the data directory.

### Required environment changes

Edit `mpc_rl/envs/velocity_tracking_env.py` only for Track B.

Required behavior:

- Keep current defaults unchanged.
- When `action_lpf_cutoff_hz` is `0` or `None`, bypass filtering so the PD target
  is exactly:

  ```python
  q_target = default_joint_pos + action_scale * action
  ```

- Add an April simple reward path selected by
  `quadruped_simple_reward_mode=april_2026`.
- The April simple reward path must use the historical terms:
  `track_lin_vel`, `track_ang_vel`, `lin_vel_forward`, `ang_vel_forward`,
  `is_terminated`, `joint_acc`, and `action_rate`.
- Do not change the current simple reward default. Existing current policies and
  experiments should continue to use the current reward.

### Required training changes

Edit `mpc_rl/train.py`:

- Define the new flags.
- Add the new values to the training config dataclass.
- Pass `quadruped_action_lpf_cutoff_hz` and `quadruped_simple_reward_mode` into
  every quadruped training, eval, checkpoint-eval, and video-eval env
  constructor.
- Pass `quadruped_mpc_replay_layout` into `PercentMPCInjectCallback`.
- Serialize the new fields into `config.json`.

### Required callback changes

Edit `mpc_rl/common/mpc_inject_callbacks.py`:

- Keep current unique batching as default.
- Add `legacy_tiled` behavior for quadruped percentage injection.
- In `legacy_tiled`, replay each quadruped torque trajectory step and tile the
  same transition across all vector-env slots before `replay_buffer.add()`,
  matching the `25e1a114` callback behavior.
- Force torque replay for April compatibility; direct saved transitions should
  not be used when `quadruped_mpc_replay_layout=legacy_tiled`.
- Preserve current direct-transition support for non-legacy runs.

### Required script changes

Update `run_quadruped_experiments.sh` for a current-branch April-compat run:

```bash
USE_GO2_SYSID="False"
DATA_DIR="data/quadruped/"
QUADRUPED_MPC_REPLAY_MODE="torque_saved_pd"
QUADRUPED_ACTION_LPF_CUTOFF_HZ=0
QUADRUPED_SIMPLE_REWARD_MODE="april_2026"
QUADRUPED_MPC_REPLAY_LAYOUT="legacy_tiled"
PERCENTAGES=(0 25 50)
SEEDS=(100 150 200)
CHECKPOINT_EVALS=(300000 400000 500000 600000)
LOG_DIR="logs/SAC-MPC-quadruped_x_vel_only_runs_min_rwrd_repro_current_legacy/"
```

Also pass every variable to `mpc_rl/train.py`:

```bash
--use_go2_sysid="${USE_GO2_SYSID}"
--quadruped_mpc_replay_mode="${QUADRUPED_MPC_REPLAY_MODE}"
--quadruped_action_lpf_cutoff_hz="${QUADRUPED_ACTION_LPF_CUTOFF_HZ}"
--quadruped_simple_reward_mode="${QUADRUPED_SIMPLE_REWARD_MODE}"
--quadruped_mpc_replay_layout="${QUADRUPED_MPC_REPLAY_LAYOUT}"
```

## Test plan for Track B

Static checks:

- Run Python compilation on touched files:

  ```bash
  python -m py_compile \
    mpc_rl/train.py \
    mpc_rl/envs/velocity_tracking_env.py \
    mpc_rl/common/mpc_inject_callbacks.py
  ```

Unit or smoke checks:

- Instantiate `QuadrupedVelocityTrackingEnv(action_lpf_cutoff_hz=0)` and verify
  one step uses the raw target without filtering.
- Instantiate the env with current defaults and verify the default LPF path still
  runs.
- Evaluate April simple reward on a fixed state/action and compare against the
  formula from `25e1a114`.
- Run a short SAC-MPC smoke command with:

  ```bash
  --total_timesteps=1000
  --num_envs=4
  --percentage=25
  --use_go2_sysid=False
  --quadruped_mpc_replay_mode=torque_saved_pd
  --quadruped_action_lpf_cutoff_hz=0
  --quadruped_simple_reward_mode=april_2026
  --quadruped_mpc_replay_layout=legacy_tiled
  ```

Acceptance checks:

- The smoke run writes a config with all legacy flags recorded.
- MPC percentage logging works.
- No direct-transition injection is used for April-compat runs.
- Current default behavior is unchanged when legacy flags are omitted.

## Recommended decision

Use Track A first. It is the only path that avoids reimplementing historical
behavior and gives the best chance of matching the April runs. Track B is useful
only if the project needs a maintainable compatibility mode on the current
branch.

If Track B is implemented, expect `mpc_rl/envs/velocity_tracking_env.py` changes.
Changing only `run_quadruped_experiments.sh` will not reproduce April behavior
because the current env and callback behavior are already different before the
script runs.

