# Reproducing MPC-Injection experiments

Use the [Conda installation](README.md#installation). All commands below run from the repository root with that environment active unless stated otherwise. Use new output directories: generators and video tools can overwrite files. `data/` holds regenerated trajectories; `logs/` holds timestamped run directories, checkpoints, normalization statistics, and evaluation videos. No download of datasets or pretrained policies is required for the basic generators.

The examples below expose existing entry points and settings; they are not a claim that full training or paper scores were rerun. Compute, storage, and training durations were not re-measured. Existing sweep scripts preserve historical research settings and should be inspected before launching a campaign.

## 2D walker

Walker walking/running use the existing velocity-only reward `(5 * move_reward + 1) / 6`, without DM Control's standing multiplier. This behavior previously lived in an edited installed package and now lives in `mpc_rl/envs/dm_control_env.py`; no package patch is needed. Standing, physics, observations, and initialization are unchanged.

Generate the existing initial-state grid (potentially a large job), then train a 25% injection example. Outputs are `data/reproduction/walker/` and timestamped runs under `logs/reproduction/walker/`.

```bash
python -m mpc_rl.planner.gen_traj_data --env walker --rollout-horizon 1000 --opt-steps 1 --output-dir data/reproduction/walker
python -m mpc_rl.train --env_name walker-walk --algorithm SAC-MPC --total_timesteps 500000 --seed 1 --percentage 25 --data_dir data/reproduction/walker --logdir logs/reproduction/walker --save_replay_buffer_checkpoints=False --save_replay_buffer_final=False
```

`--algorithm TD3-MPC` selects the other existing off-policy implementation. `--percentage 0` retains the same task/reward with no injection; 25/50/75/100 are the existing percentage-sweep values. `SAC` and `TD3` are also available without injection. The historical walker shell script contains alternate reward names in log paths; those names do not change the implemented reward.

For walker or cheetah3 evaluation, set `RUN` to the basename printed by training, keep the same training flags, and add `--play_only=True --load_run_name="$RUN"`. For example, in a new shell after setting `RUN` to your walker run:

```bash
python -m mpc_rl.train --env_name walker-walk --algorithm SAC-MPC --percentage 25 --data_dir data/reproduction/walker --logdir logs/reproduction/walker --load_run_name="$RUN" --play_only=True --num_eval_episodes 5 --num_videos 3
```

This reads the saved model/normalization and writes evaluation videos within that run directory; preserve existing outputs before rerunning.

## Go2 trotting

The original DR generator remains `gen_traj_data_mpx_dr`; it uses the Go2 MPX configuration. This small no-DR example writes trajectories to `data/reproduction/trot/`, then trains into `logs/reproduction/trot/`:

```bash
python -m mpc_rl.planner.gen_traj_data_mpx_dr --num-trajectories 10 --max-attempts 100 --episode-length 1000 --start-seed 0 --domain-rand-config-type disabled --output-dir data/reproduction/trot
python -m mpc_rl.train --env_name quadruped-velocity_tracking --algorithm SAC-MPC --percentage 25 --total_timesteps 1000000 --num_envs 4 --learning_starts 50000 --seed 1 --domain_rand=False --domain_rand_config_type disabled --data_dir data/reproduction/trot --quadruped_mpc_replay_mode direct --logdir logs/reproduction/trot --save_replay_buffer_checkpoints=False --save_replay_buffer_final=False
```

For this task, `SAC-MPC`/`TD3-MPC` select the simple velocity reward automatically; `--percentage 0` is the comparable simple-reward baseline. Plain `SAC`/`TD3` select the existing shaped-reward baseline. The default action interface is scale 0.5 with a 5-Hz target filter. Keep the data's sysID, domain-randomization, and action-interface settings aligned with training. Hardware-oriented DR presets and older torque-replay experiments remain in source; this small example is not the full hardware campaign.

## Bounding

The preserved bounding implementation is selected by `gen_traj_data_mpx_bound`, which uses `mpx.config.config_bound`; it does not replace the trotting implementation. The following controller settings come from the preserved vx=0.5 m/s milestone. Generation writes `data/reproduction/bound/` and training writes `logs/reproduction/bound/`:

```bash
python -m mpc_rl.planner.gen_traj_data_mpx_bound --num-trajectories 10 --max-attempts 100 --episode-length 1000 --start-seed 1000 --domain-rand-config-type disabled --target-command 0.5 0 0 --command-ramp-control-steps 50 --gait bound_front_first --duty-factor 0.65 --step-frequency-hz 2.5 --step-height-m 0.05 --mpx-qrot-pitch-cost 25000 --mpx-qomega-pitch-cost 1000 --action-interface mpx_bound_scale_1_no_lpf_v1 --output-dir data/reproduction/bound
python -m mpc_rl.train --env_name quadruped-velocity_tracking --algorithm SAC-MPC --percentage 25 --total_timesteps 500000 --seed 1 --domain_rand=False --domain_rand_config_type disabled --quadruped_action_interface mpx_bound_scale_1_no_lpf_v1 --quadruped_mpc_replay_mode direct --data_dir data/reproduction/bound --logdir logs/reproduction/bound --save_replay_buffer_checkpoints=False --save_replay_buffer_final=False
```

The second command is an entry-point example, not a newly established paper bounding-policy configuration. Bounding direct transitions use schema v2 and carry their action-interface descriptor. Do not train them with the default filtered scale-0.5 interface. Gait validity and full-cycle acceptance require the existing `mpx_bounding_data` validators; a three-step installation smoke only verifies schema/finite transitions. The parity tool's default report paths are tracked fixtures: always pass an explicit new `--output` path when using `check_mpx_transition_parity` (see its `--help`).

## Go2 barrel roll

Current code uses schema v4, a 125-control-step episode, and strict task/data/provenance checks. From the root, try one existing controller rollout; accepted trajectories and the manifest are written to `data/reproduction/barrel/`. A rejected attempt is recorded and may produce no accepted NPZ.

```bash
python -m mpc_rl.planner.gen_traj_data_barrel_roll --num-trajectories 1 --max-attempts 1 --start-seed 6000000 --output-dir data/reproduction/barrel
python -m mpc_rl.planner.barrel_roll_dataset data/reproduction/barrel --aggregate --demonstration-report
```

The production training contract requires a validated, atomically promoted 2,000-trajectory dataset with its summary, checksums, and `COMPLETE` marker. The retained `generate_barrel_roll_v4_production` and `run_go2_barrel_roll_v4_campaign` drivers also require historical v2/v3 checksum indexes and approved demonstration manifests, which are not bundled. Thus the original audited campaign cannot be launched unchanged from a fresh clone; do not fabricate these files or bypass its checks. Small generator and environment usage remain available.

Once the original production prerequisites are supplied and the dataset is promoted to `data/go2_barrel_roll/v4`, the frozen 25% training invocation is:

```bash
python -m mpc_rl.train --env_name quadruped-barrel_roll --algorithm SAC-MPC --percentage 25 --data_dir data/go2_barrel_roll/v4 --total_timesteps 500000 --num_envs 4 --max_episode_steps 125 --seed 1 --domain_rand=False --domain_rand_config_type disabled --use_go2_sysid=True --eval_freq 10000 --checkpoint_freq 25000 --save_replay_buffer_checkpoints=False --save_replay_buffer_final=False --logdir logs/reproduction/barrel
```

Seed 2 is the other allowed training seed. The dedicated `python -m mpc_rl.run_go2_barrel_roll_v4_0pct` launcher accepts the same arguments with `--percentage 0` for the frozen baseline. Training writes fixed-seed evaluation history and selected checkpoints into the run directory. The `evaluate_go2_barrel_roll_g9` and `evaluate_go2_barrel_roll_v4_0pct` tools retain their provenance/review checks; inspect `--help` and use explicit new output directories. Generic `--play_only` is intentionally refused for this task. Earlier G8/G9 scripts and fixtures are retained for historical compatibility, not as schema-v4 launch instructions.

## Three-legged half-cheetah

The local `cheetah3-run` environment uses `mpc_rl/tasks/cheetah/task.xml`, a 3.0 m/s speed goal, and the generator's `walker_like` initialization. Generate ten examples into `data/reproduction/cheetah3/`; train into `logs/reproduction/cheetah3/`:

```bash
python -m mpc_rl.planner.gen_traj_data_cheetah3 --num-trajectories 10 --seed-start 0 --init-mode walker_like --rollout-horizon 1000 --opt-steps 1 --output-dir data/reproduction/cheetah3
python -m mpc_rl.train --env_name cheetah3-run --algorithm SAC-MPC --percentage 25 --total_timesteps 500000 --seed 1 --cheetah3_speed_goal 3.0 --data_dir data/reproduction/cheetah3 --logdir logs/reproduction/cheetah3 --save_replay_buffer_checkpoints=False --save_replay_buffer_final=False
```

The preserved `run_cheetah_experiments.sh` uses seeds 1/50/100/150/200 and percentages 0/25/50/75/100. These are source-backed settings; the paper does not fully specify this campaign. Evaluate with the walker evaluation pattern, substituting `cheetah3-run`, its data/log directories, and `--cheetah3_speed_goal 3.0`.

## Policy export and real Go2

Start with a locally trained Go2 SAC/TD3 policy and its matching `VecNormalize` file. Set `RUN_DIR` to your run directory. From the root, the commands below write to `deploy/robots/go2/config/policy/velocity/v0/exported/` and open a simulation viewer:

```bash
python deploy/export_onnx_go2.py --model_zip "$RUN_DIR/final_model.zip" --vecnorm_pkl "$RUN_DIR/vec_normalize.pkl" --output_dir deploy/robots/go2/config/policy/velocity/v0/exported --algo SAC
python deploy/test_onnx_policy.py --onnx deploy/robots/go2/config/policy/velocity/v0/exported/policy.onnx
```

Use `--algo TD3` for TD3. Export bakes observation normalization into the ONNX graph and numerically compares it with the Python policy. Test the exact exported policy in simulation before robot use. The checked-in deployment template uses the original trotting action scale/filter and FL/FR/RL/RR policy joint order; exporting alone does not make bounding or barrel-roll policies compatible with that hardware template.

Native prerequisites: Boost program_options, yaml-cpp, zlib, Eigen3, fmt, installed Unitree SDK2 and CycloneDDS, and native **ONNX Runtime 1.22.0** for the host architecture. CMake expects its libraries under `deploy/thirdparty/onnxruntime-linux-x64-1.22.0/lib/` or `onnxruntime-linux-aarch64-1.22.0/lib/`; obtain the matching official release archive. Headers/notices are retained, binaries are not bundled. The Python ONNX Runtime version in Conda is separate. From the root, build outputs go to `deploy/robots/go2/build/`:

```bash
cmake -S deploy/robots/go2 -B deploy/robots/go2/build -DCMAKE_BUILD_TYPE=Release
cmake --build deploy/robots/go2/build --target go2_ctrl go2_torque_recorder_test --parallel 2
```

Hardware operation was not revalidated during cleanup. Use a harness and a direct Ethernet connection (host address example `192.168.123.99/24`). Run only one low-level controller. `MotionSwitcherClient` releases Unitree's active `sport_mode`/`ai_sport`/`advanced_sport` service; keep `basic_service` enabled. If another `rt/lowcmd` publisher is detected, stop it and investigate before proceeding. Never run the Go2 controller alongside an AMP/ROS/SDK low-level controller.

For handoff, use the robot controller's `L2+B`, then `L2+R2` debug mode. Set `NETWORK_INTERFACE` to your connected Ethernet interface and `TORQUE_OUTPUT` to a new absolute `.npz` path. From `deploy/robots/go2/build`, launch the locally exported policy:

```bash
./go2_ctrl --network="$NETWORK_INTERFACE" --policy_dir=config/policy/velocity/v0 --torque_output "$TORQUE_OUTPUT"
```

The application starts in Passive. `L2+Up` enters FixStand, `R2+A` enters Velocity, and `L2+B` returns to Passive as the abort action. Passive is **damping, not zero torque** (`kp=0`, `kd=3`, current motor positions). In keyboard input mode, Up/Down change forward velocity by 0.1 and `R` sets it to zero; the gamepad is still required for transitions and aborts. Stop on rattling, rapid oscillation, or repeated low-command warnings. These procedures provide no safety guarantee.

Optional torque recording stores low-state `tau_est`, not the zero feed-forward torque command, in FL/FR/RL/RR training order. `tau_applied` is the archive's compatibility key. Recording starts in policy mode and is finalized on Passive, exit, or Ctrl+C; no archive is written if policy never starts. Reusing the path for another activation overwrites it.

The external AMP-MPC launcher mentioned in historical hardware notes is not bundled here. Its Space/5 command only zeros velocity; Q/Esc sends damping and exits. Those keys are not the Go2 C++ application's controls.

## Validation and reproduction limits

Cleanup checks cover disposable Conda installation, imports outside the checkout, focused tests, numerical comparisons against preserved implementations, five short task startups, tiny trajectory generation, temporary-policy ONNX export, and native compilation. Full training, paper scores, physical hardware, and an aarch64 build were not rerun. Public recursive retrieval of the integrated MPX commit remains a publication-stage check.

The source controller costs do not exactly match the paper's Appendix B table; existing Go2 and bounding values are preserved separately. The repository does not contain enough evidence to reconcile every paper campaign setting. The old dataset-schema contrast test requires an unbundled historical dataset; the optional sysID HTML comparison also expects an unbundled `deploy/sys_id/report.html`, while runtime sysID uses the checked-in numerical table. Retained JSON files under `docs/mpx_bound_milestone_reports/` are test/parity contracts, not regenerated release results. Use explicit test paths; several files under `tests/` are interactive scripts rather than pytest suites.
