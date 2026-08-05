# Quadruped April 2026 Training Reproduction Diagnosis

## Executive summary

The April 4-5, 2026 runs in `logs/SAC-MPC-quadruped_x_vel_only_runs_min_rwrd/`
were not produced by the current quadruped training code path. Their saved
configs match the common hyperparameters in the April-era
`run_quadruped_experiments.sh`, but several behavioral defaults and
implementation details changed after those runs.

The most likely source commit for the April runs is `25e1a114` from April 3,
2026. It is the last relevant commit before the run timestamps and it matches
the April configs for algorithm, seeds, percentages, `num_envs=4`,
`data_dir=data/quadruped/`, disabled DR, buffer size, batch size, learning
starts, and checkpoint eval steps. The script's log directory was presumably
edited locally to `logs/SAC-MPC-quadruped_x_vel_only_runs_min_rwrd/`, because the
committed script at `25e1a114` points to `logs/SAC-MPC-quadruped_x_vel_only_runs/`.

The reproduction attempts are therefore different experiments even if the
visible hyperparameters look similar. The important differences are:

1. Current training enables Go2 sysID dynamics by default; April did not have
   this code or flag.
2. Current env actions are low-pass filtered before PD; April applied action
   targets directly.
3. The SAC-MPC "simple reward" used for quadruped MPC injection changed after
   the April runs.
4. Quadruped MPC injection changed from tiling one demo transition across all
   vector-env slots to packing unique demo transitions into those slots.
5. Newer runs use different MPC data formats/direct-transition behavior in
   `data/quadruped_dr/sysid_nominal/`, and the `data/quadruped/` legacy files
   lack direct-transition and sysID metadata.
6. The May `_ablation` directory contains interrupted runs, and the May
   `_ablations` directory has no `config.json` files, so neither is a clean
   reproduction set.

## Artifact timeline

Original April runs:

- Directory: `logs/SAC-MPC-quadruped_x_vel_only_runs_min_rwrd/`
- Run timestamps: April 4, 2026 23:28 through April 5, 2026 15:32.
- Seeds: `100`, `150`, `200`.
- Percentages: `0`, `25`, `50`.
- All nine runs have `config.json`, tensorboard events, checkpoints, final model,
  replay buffer, and vec-normalize artifacts.
- The final plot PNG in this directory is timestamped April 28, 2026, so it was
  generated later from the saved April events.

Newer reproduction artifacts:

- `logs/SAC-MPC-quadruped_x_vel_only_runs_min_rwrd_ablation/`
  has `config.json` files, but several runs did not finish. The 25% seed 100 and
  25% seed 200 runs stopped at about 57k and 120k steps, and the 75% seed 300
  run stopped at about 410k steps. They lack `final_model.zip` and
  `replay_buffer.pkl`.
- `logs/SAC-MPC-quadruped_x_vel_only_runs_min_rwrd_ablations/`
  has tensorboard event files and a plot PNG, but no `config.json` files. Its
  settings cannot be proven from artifacts.

## Config comparison

The April configs all contain only the older fields:

- `algorithm=SAC-MPC`
- `total_timesteps=1000000`
- `num_envs=4`
- `learning_starts=50000`
- `buffer_size=5000000`
- `learning_rate=0.0003`
- `batch_size=256`
- `policy_delay=2`
- `data_dir=data/quadruped/`
- `domain_randomization.enabled=false`
- `checkpoint_evals=[300000,400000,500000,600000]`

The May `_ablation` configs add behavior-bearing fields that April did not have:

- `use_go2_sysid=true`
- `quadruped_mpc_replay_mode=direct`
- expanded domain-randomization metadata
- `checkpoint_evals=[500000]`

`use_go2_sysid=true` is not just metadata. In current `mpc_rl/train.py`, the
flag defaults to true and is passed into every quadruped env. In April
`25e1a114`, `mpc_rl/train.py` had no `use_go2_sysid` flag at all.

## Git-derived source of the April behavior

The run window is after:

- `25e1a114` on April 3, 2026: "mpc-injection working for only x velocity
  commands..."

and before:

- `4f70c650` on April 6, 2026: changed the simple reward to remove angular,
  joint-acceleration, and action-rate terms.
- `e073aa2b` on April 6, 2026: changed quadruped MPC injection tiling so unique
  trajectories are inserted across replay rows.

That makes `25e1a114` the best source snapshot for the April training logic.

## Behavioral differences that explain the mismatch

### 1. Go2 sysID dynamics are new and enabled in current runs

Current code applies identified Go2 joint dynamics when `use_go2_sysid=True`.
This writes per-joint `armature`, `damping`, and `frictionloss` into the MuJoCo
model. The source file `mpc_rl/envs/go2_sysid.py` did not exist at `25e1a114`;
it was introduced after the April runs.

This affects both pure RL and MPC-injection runs. For `data/quadruped/`, the MPC
files were generated April 1-2 and contain only legacy torque-replay arrays:
`qpos`, `qvel`, `tau_applied`, `commands`, `default_joint_pos`, `sim_dt`, and
`control_dt`. They do not contain direct `policy_obs/actions/rewards` or sysID
metadata. Replaying those old torques in a sysID-patched env is not the same
plant that produced the April demos.

### 2. Current env filters actions; April env did not

At `25e1a114`, `step()` computed:

```python
q_target = self.default_joint_pos + self.action_scale * action
```

Current `velocity_tracking_env.py` computes a raw target and then applies a
5 Hz low-pass filter before PD:

```python
self._raw_q_target = self.default_joint_pos + self.action_scale * action
q_target = self._apply_action_lpf(self._raw_q_target)
```

This changes the action-to-torque dynamics for every environment transition.
Even a 0% MPC run is therefore not a reproduction under current code.

### 3. The simple reward changed after the April runs

At `25e1a114`, `_compute_simple_reward()` used `self.reward_cfg` and included:

- `track_lin_vel`
- `track_ang_vel`
- `lin_vel_forward`
- `ang_vel_forward`
- termination cost
- joint acceleration penalty
- action-rate penalty

Current code uses `_SIMPLE_REWARD_CFG` and comments out the angular tracking,
angular forward, joint acceleration, and action-rate terms in the actual reward
sum. This change was introduced after the April runs (`4f70c650` on April 6).

This also explains why comparing raw `rollout/ep_rew_mean` across April and May
is partially misleading: the reward scale and objective changed.

### 4. The quadruped MPC injection semantics changed

At `25e1a114`, quadruped injection replayed one demo control step and tiled that
same transition across all vector-env slots before `ReplayBuffer.add()`. With
`num_envs=4`, each unique MPC transition became four counted replay-buffer
transitions.

Current code accumulates unique quadruped transitions and flushes them in
`n_envs`-sized batches. The replay-buffer percentage may look similar, but the
actual diversity and row layout of MPC data are different. This affects every
nonzero-percentage ablation.

This change landed after the April runs (`e073aa2b`, April 6).

### 5. `quadruped_mpc_replay_mode` is not actually passed by the current script

The current working copy of `run_quadruped_experiments.sh` defines:

```bash
QUADRUPED_MPC_REPLAY_MODE="torque_current_pd"
```

but the `python mpc_rl/train.py ...` command does not pass
`--quadruped_mpc_replay_mode`. Consequently the May `_ablation` configs show the
train default:

```json
"quadruped_mpc_replay_mode": "direct"
```

For legacy `data/quadruped/` files, "direct" falls back to torque replay because
the direct-transition keys are absent. For newer files that do contain direct
transition keys, "direct" injects saved observations/actions/rewards instead of
replaying torques through the current env.

### 6. The two newer data directories are materially different

`data/quadruped/`:

- 9,113 `.npz` files.
- File mtimes are April 1-2, 2026.
- Legacy format only: no saved `policy_obs`, `actions`, `rewards`, or
  `terminated_ctrl`.

`data/quadruped_dr/sysid_nominal/`:

- 7,609 `.npz` files.
- File mtimes start April 24, 2026.
- Contains direct RL transitions: `policy_obs`, `next_policy_obs`,
  `privileged_obs`, `next_privileged_obs`, `actions`, `rewards`,
  `terminated_ctrl`, plus DR patch metadata.

The May `_ablations` tensorboard directory likely comes from the later direct
transition workflow, not the April torque-replay workflow. Because it has no
configs, this remains an inference from file names, timestamps, and git state.

## TensorBoard evidence

April original runs, final `rollout/ep_rew_mean`:

| pct | seeds | final mean | notes |
| --- | --- | ---: | --- |
| 0 | 100,150,200 | 1639.7 | all complete |
| 25 | 100,150,200 | 1223.3 | all complete |
| 50 | 100,150,200 | 1516.1 | all complete |

May `_ablation` runs, final `rollout/ep_rew_mean`:

| pct | available seeds | final mean | notes |
| --- | --- | ---: | --- |
| 0 | 100,200,300 | 1207.8 | complete, but seed100 collapsed late |
| 25 | 100,200,300 | 623.5 | seed100 and seed200 stopped early |
| 50 | 300 | 1170.1 | only seed300 present and complete |
| 75 | 300 | 886.2 | stopped early |

May `_ablations` event-only runs, final `rollout/ep_rew_mean`:

| pct | seeds | final mean | notes |
| --- | --- | ---: | --- |
| 0 | 1,50,100,150,200 | 1725.0 | no configs |
| 25 | 1,50,100,150,200 | 1488.9 | no configs |
| 50 | 1,50,100,150,200 | 1094.7 | no configs |
| 75 | 1,50,100,150,200 | 927.6 | no configs |
| 100 | 1,50,100,150,200 | 345.0 | no configs; one run stopped at ~150k |

The 25% and 50% trends differ across directories, but these are not controlled
replications: code, data format, env dynamics, reward, injection semantics, and
in some cases run completeness differ.

## Why matching `config.json` was insufficient

The April configs did not serialize all behavior-bearing choices, because those
choices did not exist yet. In particular:

- no `use_go2_sysid`
- no `quadruped_mpc_replay_mode`
- no action LPF parameter
- no marker for old tiled-vs-unique quadruped replay-buffer insertion
- no serialized reward implementation version
- no git commit hash
- no MPC data file manifest or data generation commit

So a current run can match the old JSON fields and still run different code.

## Bottom line

The April results are reproducible only as an April-code reproduction, not as a
current-code rerun with the same visible hyperparameters. The closest target is:

- source around `25e1a114`
- `data_dir=data/quadruped/`
- no Go2 sysID dynamics
- no action low-pass filter
- old simple reward with angular, joint-acceleration, and action-rate terms
- old quadruped MPC injection that tiled one demo step across vector-env slots
- `num_envs=4`
- seeds `100 150 200`
- percentages `0 25 50`
- checkpoint evals `300000,400000,500000,600000`

Running current `run_quadruped_experiments.sh` does not satisfy those conditions.
The current script also defines `QUADRUPED_MPC_REPLAY_MODE` without passing it,
so the intended replay mode is not reflected in actual runs unless the flag is
added to the command.
