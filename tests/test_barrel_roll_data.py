from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium import spaces
from scipy.spatial.transform import Rotation

from mpc_rl.envs.barrel_roll_common import (
    CONTROL_DT,
    CONTROL_STEPS,
    REWARD_CONFIG,
    ROLL_DIRECTION_SIGN,
    ROLL_END_TIME,
    ROLL_START_TIME,
    SIM_DT,
    desired_roll_at_time,
    maneuver_phase_at_time,
    reward_config_dict,
    success_config_dict,
)
from mpc_rl.planner.barrel_roll_dataset import (
    ACTION_SEMANTICS,
    CONTACT_CONVENTION,
    CONTROLLER_MODE,
    DOMAIN_RANDOMIZATION,
    MPC_DT,
    MPC_NODES,
    MPC_STATE_DIM,
    PD_KD,
    PD_KP,
    PHYSICS_STEPS,
    REPLANNING_FREQUENCY_HZ,
    ROBOT_ID,
    TASK_ID,
    TORQUE_LIMITS,
    TRACKING_KD,
    TRACKING_KP,
    WARM_START_POLICY,
    aggregate_barrel_roll_dataset,
    atomic_save_npz,
    canonical_json,
    expected_effective_config,
    schedule_dict,
    sha256_file,
    validate_barrel_roll_file,
)
from mpc_rl.common.mpc_inject_callbacks import PercentMPCInjectCallback
from mpc_rl.common.tagged_dict_replay_buffer import TaggedDictReplayBuffer

_INTEGRITY_SPEC = importlib.util.spec_from_file_location(
    "check_data_integrity",
    Path(__file__).resolve().parents[1] / "utils" / "check_data_integrity.py",
)
assert _INTEGRITY_SPEC is not None and _INTEGRITY_SPEC.loader is not None
_INTEGRITY_MODULE = importlib.util.module_from_spec(_INTEGRITY_SPEC)
_INTEGRITY_SPEC.loader.exec_module(_INTEGRITY_MODULE)
check_directory = _INTEGRITY_MODULE.check_directory
check_npz_file = _INTEGRITY_MODULE.check_npz_file


def _valid_arrays(seed: int = 0) -> dict[str, np.ndarray]:
    physics_time = np.arange(PHYSICS_STEPS + 1, dtype=np.float64) * SIM_DT
    control_time = np.arange(1, CONTROL_STEPS + 1, dtype=np.float64) * CONTROL_DT
    roll = np.array([desired_roll_at_time(time_s) for time_s in physics_time])
    qpos = np.zeros((19, PHYSICS_STEPS + 1), dtype=np.float64)
    qpos[2] = 0.27
    qpos[3] = np.cos(roll / 2.0)
    qpos[4] = np.sin(roll / 2.0)
    qvel = np.zeros((18, PHYSICS_STEPS + 1), dtype=np.float64)

    policy_all = np.zeros((CONTROL_STEPS + 1, 45), dtype=np.float64)
    for index in range(CONTROL_STEPS + 1):
        time_s = index * CONTROL_DT
        desired = desired_roll_at_time(time_s)
        policy_all[index, 6:9] = (
            maneuver_phase_at_time(time_s),
            np.sin(desired),
            np.cos(desired),
        )
    privileged_all = np.zeros((CONTROL_STEPS + 1, 4), dtype=np.float64)
    privileged_all[:, 3] = roll[::4]

    foot_contacts = np.zeros((PHYSICS_STEPS, 4), dtype=bool)
    foot_contacts[-20:] = True
    stable_streak = np.zeros(CONTROL_STEPS, dtype=np.int64)
    stable_streak[-5:] = np.arange(1, 6)
    classifier = np.zeros(CONTROL_STEPS, dtype=bool)
    classifier[-1] = True

    desired_ctrl = np.array([desired_roll_at_time(time_s) for time_s in control_time])
    pre_roll = roll[:-1:4]
    post_roll = roll[4::4]
    expected_step = 2.0 * np.pi * CONTROL_DT / (ROLL_END_TIME - ROLL_START_TIME)
    rewards = REWARD_CONFIG.tracking_weight * np.exp(
        -((desired_ctrl - post_roll) / REWARD_CONFIG.tracking_sigma) ** 2
    )
    rewards += REWARD_CONFIG.progress_weight * np.clip(
        ROLL_DIRECTION_SIGN * (post_roll - pre_roll) / expected_step,
        -REWARD_CONFIG.progress_clip,
        REWARD_CONFIG.progress_clip,
    )
    rewards[-1] += REWARD_CONFIG.success_bonus

    terminated = np.zeros(CONTROL_STEPS, dtype=bool)
    terminated[-1] = True
    zeros_tau = np.zeros((12, PHYSICS_STEPS), dtype=np.float64)
    effective_json = canonical_json(expected_effective_config())
    final_roll, final_pitch, _ = Rotation.from_quat(
        np.roll(qpos[3:7, -1], -1)
    ).as_euler("xyz")
    arrays = {
        "policy_obs": policy_all[:-1],
        "next_policy_obs": policy_all[1:],
        "privileged_obs": privileged_all[:-1],
        "next_privileged_obs": privileged_all[1:],
        "actions": np.zeros((CONTROL_STEPS, 12), dtype=np.float64),
        "rewards": rewards,
        "terminated_ctrl": terminated,
        "truncated_ctrl": np.zeros(CONTROL_STEPS, dtype=bool),
        "qpos": qpos,
        "qvel": qvel,
        "tau_applied": zeros_tau.copy(),
        "tau_mpx": zeros_tau.copy(),
        "tau_raw": zeros_tau.copy(),
        "q_des": zeros_tau.copy(),
        "dq_des": zeros_tau.copy(),
        "X_updates": np.zeros(
            (CONTROL_STEPS, MPC_NODES + 1, MPC_STATE_DIM), dtype=np.float32
        ),
        "U_updates": np.zeros((CONTROL_STEPS, MPC_NODES, 12), dtype=np.float32),
        "physics_time": physics_time,
        "control_time": control_time,
        "phase_ctrl": np.array([maneuver_phase_at_time(time_s) for time_s in control_time]),
        "desired_roll_ctrl": desired_ctrl,
        "measured_roll_physics": roll,
        "foot_contacts": foot_contacts,
        "nonfoot_contact": np.full(PHYSICS_STEPS, "", dtype="<U96"),
        "stable_contact_streak": stable_streak,
        "classifier_result": classifier,
        "residual_actions_unclipped": np.zeros((CONTROL_STEPS, 12)),
        "action_clipped": np.zeros((CONTROL_STEPS, 12), dtype=bool),
        "mpx_saturation_by_actuator": np.zeros((12, PHYSICS_STEPS), dtype=bool),
        "applied_saturation_by_actuator": np.zeros((12, PHYSICS_STEPS), dtype=bool),
        "solve_seconds": np.full(CONTROL_STEPS, 0.01),
        "solve_iterations": np.concatenate(([100], np.ones(CONTROL_STEPS - 1))).astype(np.int64),
        "solve_iteration_limit": np.concatenate(([100], np.ones(CONTROL_STEPS - 1))).astype(np.int64),
        "solve_objective_norm_sq": np.ones(CONTROL_STEPS),
        "solve_constraint_norm_sq": np.ones(CONTROL_STEPS),
        "solve_finite": np.ones(CONTROL_STEPS, dtype=bool),
        "solve_replanned": np.ones(CONTROL_STEPS, dtype=bool),
        "solve_phase_index": np.arange(CONTROL_STEPS, dtype=np.int64) * 2,
        "solve_elapsed_time": np.arange(CONTROL_STEPS) * CONTROL_DT,
        "warm_start_shift": np.concatenate(([0], np.full(CONTROL_STEPS - 1, 2))).astype(np.int64),
        "default_joint_pos": np.tile(np.array([0.0, 0.9, -1.8]), 4),
        "pd_kp": PD_KP.copy(),
        "pd_kd": PD_KD.copy(),
        "tracking_kp": TRACKING_KP.copy(),
        "tracking_kd": TRACKING_KD.copy(),
        "torque_limits": TORQUE_LIMITS.copy(),
        "schema_version": np.asarray(1),
        "task_id": np.asarray(TASK_ID),
        "robot_id": np.asarray(ROBOT_ID),
        "roll_direction": np.asarray(ROLL_DIRECTION_SIGN),
        "rollout_seed": np.asarray(seed),
        "sampled_spread": np.asarray(0.05),
        "success": np.asarray(True),
        "failure_reason": np.asarray(""),
        "domain_randomization": np.asarray(DOMAIN_RANDOMIZATION),
        "go2_sysid_enabled": np.asarray(True),
        "sim_dt": np.asarray(SIM_DT),
        "control_dt": np.asarray(CONTROL_DT),
        "decimation": np.asarray(4),
        "controller_steps": np.asarray(CONTROL_STEPS),
        "physics_steps": np.asarray(PHYSICS_STEPS),
        "maneuver_horizon": np.asarray(1.4),
        "mpc_dt": np.asarray(MPC_DT),
        "replanning_frequency_hz": np.asarray(REPLANNING_FREQUENCY_HZ),
        "action_scale": np.asarray(0.5),
        "action_lpf_cutoff_hz": np.asarray(5.0),
        "action_lpf_alpha": np.asarray(1.0 - np.exp(-2.0 * np.pi * 5.0 * CONTROL_DT)),
        "action_semantics": np.asarray(ACTION_SEMANTICS),
        "saved_action_reproduces_lpf_transition": np.asarray(False),
        "contact_convention": np.asarray(CONTACT_CONVENTION),
        "controller_mode": np.asarray(CONTROLLER_MODE),
        "warm_start_policy": np.asarray(WARM_START_POLICY),
        "schedule_json": np.asarray(canonical_json(schedule_dict())),
        "success_config_json": np.asarray(canonical_json(success_config_dict())),
        "reward_config_json": np.asarray(canonical_json(reward_config_dict())),
        "effective_config_json": np.asarray(effective_json),
        "effective_config_sha256": np.asarray(hashlib.sha256(effective_json.encode()).hexdigest()),
        "root_commit": np.asarray("0" * 40),
        "mpx_commit": np.asarray("1" * 40),
        "solver_commit": np.asarray("2" * 40),
        "gym_quadruped_commit": np.asarray("3" * 40),
        "root_worktree_dirty": np.asarray(True),
        "mpx_worktree_dirty": np.asarray(False),
        "mpx_xml_sha256": np.asarray("4" * 64),
        "rollout_xml_sha256": np.asarray("5" * 64),
        "generator_source_sha256": np.asarray("6" * 64),
        "generator_command": np.asarray("python -m mpc_rl.planner.gen_traj_data_barrel_roll"),
        "generator_config_json": np.asarray(canonical_json({"seed": seed})),
        "runtime_versions_json": np.asarray(canonical_json({"python": "test"})),
        "final_roll": np.asarray(final_roll),
        "final_pitch": np.asarray(final_pitch),
        "final_base_height": np.asarray(qpos[2, -1]),
        "final_roll_progress": np.asarray(roll[-1]),
        "action_clip_fraction": np.asarray(0.0),
        "mpx_torque_saturation_fraction": np.asarray(0.0),
        "torque_saturation_fraction": np.asarray(0.0),
    }
    return {key: np.asarray(value) for key, value in arrays.items()}


def _write_valid(path: Path, seed: int = 0) -> Path:
    np.savez_compressed(path, **_valid_arrays(seed))
    return path


class _DummyLogger:
    def record(self, *args, **kwargs):
        del args, kwargs


class _DummyCallbackModel:
    def __init__(self, replay_buffer, num_envs):
        self.replay_buffer = replay_buffer
        self._env = SimpleNamespace(num_envs=num_envs)
        self.logger = _DummyLogger()

    def get_env(self):
        return self._env


def _barrel_replay_buffer(*, n_envs: int, buffer_size: int = 512):
    return TaggedDictReplayBuffer(
        buffer_size=buffer_size,
        observation_space=spaces.Dict(
            {
                "policy": spaces.Box(-np.inf, np.inf, shape=(45,), dtype=np.float64),
                "privileged": spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float64),
            }
        ),
        action_space=spaces.Box(-1.0, 1.0, shape=(12,), dtype=np.float64),
        device="cpu",
        n_envs=n_envs,
        optimize_memory_usage=False,
        handle_timeout_termination=True,
    )


def test_valid_schema_loads_without_pickle_and_reports_saturation(tmp_path):
    path = _write_valid(tmp_path / "go2_barrel_roll_v1_dir_pos_seed_000000_ep_070.npz")
    with np.load(path, allow_pickle=False) as data:
        assert data["policy_obs"].shape == (70, 45)
        assert all(data[key].dtype.kind != "O" for key in data.files)
    report = validate_barrel_roll_file(path)
    assert report.valid, report.errors
    assert report.metrics == {
        "action_clip_fraction": 0.0,
        "mpx_torque_saturation_fraction": 0.0,
        "torque_saturation_fraction": 0.0,
    }


def test_atomic_save_leaves_only_valid_final_archive(tmp_path):
    path = tmp_path / "go2_barrel_roll_v1_atomic.npz"
    atomic_save_npz(path, _valid_arrays())
    assert validate_barrel_roll_file(path).valid
    assert list(tmp_path.iterdir()) == [path]


def _wrong_schema(arrays):
    arrays["schema_version"] = np.asarray(2)


def _wrong_task(arrays):
    arrays["task_id"] = np.asarray("quadruped_velocity_tracking")


def _wrong_dimensions(arrays):
    arrays["actions"] = arrays["actions"][:-1]


def _wrong_timing(arrays):
    arrays["sim_dt"] = np.asarray(0.01)


def _non_finite(arrays):
    arrays["qvel"][0, 0] = np.nan


def _invalid_action(arrays):
    arrays["actions"][0, 0] = 1.01


def _broken_adjacency(arrays):
    arrays["next_policy_obs"] = arrays["next_policy_obs"].copy()
    arrays["next_policy_obs"][0, 0] = 1.0


def _wrong_phase(arrays):
    arrays["phase_ctrl"][10] += 0.1


def _wrong_reward(arrays):
    arrays["rewards"][20] += 1.0


def _wrong_done(arrays):
    arrays["terminated_ctrl"][-1] = False


def _nonfoot_contact(arrays):
    arrays["nonfoot_contact"][0] = "torso"


def _incomplete_rotation(arrays):
    arrays["qpos"][3:7] = np.array([[1.0], [0.0], [0.0], [0.0]])


def _unstable_landing(arrays):
    arrays["foot_contacts"][-20:] = False


def _wrong_reward_config(arrays):
    config = reward_config_dict()
    config["success_bonus"] = 9.0
    arrays["reward_config_json"] = np.asarray(canonical_json(config))


@pytest.mark.parametrize(
    "mutation",
    [
        _wrong_schema,
        _wrong_task,
        _wrong_dimensions,
        _wrong_timing,
        _non_finite,
        _invalid_action,
        _broken_adjacency,
        _wrong_phase,
        _wrong_reward,
        _wrong_done,
        _nonfoot_contact,
        _incomplete_rotation,
        _unstable_landing,
        _wrong_reward_config,
    ],
    ids=lambda mutation: mutation.__name__.removeprefix("_"),
)
def test_required_corruptions_are_rejected(tmp_path, mutation):
    arrays = _valid_arrays()
    mutation(arrays)
    path = tmp_path / "go2_barrel_roll_v1_corrupt.npz"
    np.savez_compressed(path, **arrays)
    report = validate_barrel_roll_file(path)
    assert not report.valid
    assert report.errors


def test_integrity_checker_delegates_barrel_files_and_preserves_generic_npz(tmp_path):
    _write_valid(tmp_path / "go2_barrel_roll_v1_valid.npz")
    corrupted = _valid_arrays()
    corrupted["task_id"] = np.asarray("velocity_tracking")
    np.savez_compressed(tmp_path / "go2_barrel_roll_v1_wrong_task.npz", **corrupted)
    generic = tmp_path / "quadruped_velocity_data.npz"
    np.savez_compressed(generic, qpos=np.zeros((2, 3)))
    assert check_npz_file(generic) == (True, None)
    valid, invalid, failures = check_directory(tmp_path, verbose=False)
    assert (valid, invalid) == (2, 1)
    assert failures[0][0].name == "go2_barrel_roll_v1_wrong_task.npz"


def test_integrity_cli_exits_nonzero_for_barrel_corruption(tmp_path):
    arrays = _valid_arrays()
    arrays["rewards"][0] += 1.0
    path = tmp_path / "go2_barrel_roll_v1_corrupt.npz"
    np.savez_compressed(path, **arrays)
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "utils" / "check_data_integrity.py"),
            str(path),
            "--quiet",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "rewards mismatch" in result.stdout


def test_aggregate_writes_manifest_checksums_and_summary(tmp_path):
    paths = [
        _write_valid(
            tmp_path / f"go2_barrel_roll_v1_dir_pos_seed_{seed:06d}_ep_070.npz",
            seed,
        )
        for seed in (0, 1)
    ]
    records = [
        {
            "accepted": True,
            "failure_reason": None,
            "seed": seed,
            "trajectory_file": path.name,
        }
        for seed, path in enumerate(paths)
    ]
    (tmp_path / "generation_manifest_worker0.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = aggregate_barrel_roll_dataset(tmp_path)
    assert summary["file_count"] == 2
    assert summary["transition_count"] == 140
    assert summary["seeds"] == [0, 1]
    checksum_lines = (tmp_path / "checksums.sha256").read_text().splitlines()
    assert checksum_lines == [f"{sha256_file(path)}  {path.name}" for path in paths]
    saved_summary = json.loads((tmp_path / "dataset_summary.json").read_text())
    assert saved_summary["checksum_index_sha256"] == sha256_file(
        tmp_path / "checksums.sha256"
    )
    assert len((tmp_path / "generation_manifest_aggregate.jsonl").read_text().splitlines()) == 2


def test_barrel_injection_is_strict_sorted_direct_multi_env_and_exact_25_percent(
    tmp_path, monkeypatch
):
    selected = _write_valid(
        tmp_path / "b_go2_barrel_roll_v1_dir_pos_seed_000001_ep_070.npz",
        seed=1,
    )
    first = _write_valid(
        tmp_path / "a_go2_barrel_roll_v1_dir_pos_seed_000000_ep_070.npz",
        seed=0,
    )
    callback = PercentMPCInjectCallback(
        domain="quadruped",
        task="barrel_roll",
        target_percentage=25,
        data_dir=str(tmp_path),
        random_select=False,
        trajectory_files=[selected.name],
        expected_quadruped_task=TASK_ID,
        expected_quadruped_schema_version=1,
        quadruped_mpc_replay_mode="direct",
        verbose=0,
    )
    assert callback.available_files == [first, selected]

    replay_buffer = _barrel_replay_buffer(n_envs=2)
    zero_obs = {
        "policy": np.zeros((2, 45)),
        "privileged": np.zeros((2, 4)),
    }
    for _ in range(105):
        replay_buffer.add(
            obs=zero_obs,
            next_obs=zero_obs,
            action=np.zeros((2, 12)),
            reward=np.zeros(2),
            done=np.zeros(2),
            infos=[{}, {}],
            source=0,
        )

    captured_infos = []
    original_add = replay_buffer.add

    def _capture_add(*args, **kwargs):
        captured_infos.extend(kwargs["infos"])
        return original_add(*args, **kwargs)

    monkeypatch.setattr(replay_buffer, "add", _capture_add)
    callback.init_callback(_DummyCallbackModel(replay_buffer, num_envs=2))
    monkeypatch.setattr(
        callback,
        "_replay_quadruped_trajectory",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("barrel data must not construct or use velocity torque replay")
        ),
    )
    callback._inject_mpc_trajectories()

    assert replay_buffer.get_mpc_percentage() == pytest.approx(25.0)
    assert replay_buffer.size() == 140
    assert np.all(replay_buffer.transition_sources[105:140] == 1)
    with np.load(selected, allow_pickle=False) as data:
        injected_policy = replay_buffer.observations["policy"][105:140].reshape(70, 45)
        injected_privileged = replay_buffer.observations["privileged"][105:140].reshape(70, 4)
        injected_actions = replay_buffer.actions[105:140].reshape(70, 12)
        injected_rewards = replay_buffer.rewards[105:140].reshape(70)
        injected_dones = replay_buffer.dones[105:140].reshape(70)
        np.testing.assert_array_equal(injected_policy, data["policy_obs"])
        np.testing.assert_array_equal(injected_privileged, data["privileged_obs"])
        np.testing.assert_array_equal(injected_actions, data["actions"].astype(np.float32))
        np.testing.assert_array_equal(injected_rewards, data["rewards"].astype(np.float32))
        np.testing.assert_array_equal(injected_dones, data["terminated_ctrl"])
    assert captured_infos[-1]["is_success"] is True
    assert captured_infos[-1]["failure_reason"] is None
    assert captured_infos[-1]["TimeLimit.truncated"] is False
    assert replay_buffer.timeouts[139, 1] == 0.0


def test_barrel_injection_rejects_any_malformed_file_without_fallback(tmp_path):
    _write_valid(tmp_path / "go2_barrel_roll_v1_valid.npz")
    malformed = _valid_arrays(seed=2)
    malformed["actions"][0, 0] = 2.0
    np.savez_compressed(tmp_path / "go2_barrel_roll_v1_malformed.npz", **malformed)

    with pytest.raises(ValueError, match="invalid barrel-roll trajectory"):
        PercentMPCInjectCallback(
            domain="quadruped",
            task="barrel_roll",
            target_percentage=25,
            data_dir=str(tmp_path),
            expected_quadruped_task=TASK_ID,
            expected_quadruped_schema_version=1,
            quadruped_mpc_replay_mode="direct",
            verbose=0,
        )


def test_barrel_task_rejects_explicit_torque_replay(tmp_path):
    _write_valid(tmp_path / "go2_barrel_roll_v1_valid.npz")
    with pytest.raises(ValueError, match="requires direct replay"):
        PercentMPCInjectCallback(
            domain="quadruped",
            task="barrel_roll",
            data_dir=str(tmp_path),
            quadruped_mpc_replay_mode="torque_saved_pd",
            verbose=0,
        )


def test_direct_transition_timeout_sets_done_and_sb3_timeout_info():
    callback = PercentMPCInjectCallback(
        domain="quadruped",
        task="velocity_tracking",
        target_percentage=100,
        verbose=0,
    )
    replay_buffer = _barrel_replay_buffer(n_envs=1)
    captured_infos = []
    original_add = replay_buffer.add

    def _capture_add(*args, **kwargs):
        captured_infos.extend(kwargs["infos"])
        return original_add(*args, **kwargs)

    replay_buffer.add = _capture_add
    callback.init_callback(_DummyCallbackModel(replay_buffer, num_envs=1))
    trajectory = {
        "policy_obs": np.zeros((1, 45)),
        "next_policy_obs": np.ones((1, 45)),
        "privileged_obs": np.zeros((1, 4)),
        "next_privileged_obs": np.ones((1, 4)),
        "actions": np.zeros((1, 12)),
        "rewards": np.ones(1),
        "terminated_ctrl": np.zeros(1, dtype=bool),
        "truncated_ctrl": np.ones(1, dtype=bool),
    }
    callback._inject_saved_quadruped_transitions(trajectory)

    assert replay_buffer.dones[0, 0] == 1.0
    assert replay_buffer.timeouts[0, 0] == 1.0
    assert captured_infos[0]["TimeLimit.truncated"] is True


def test_barrel_terminal_failure_is_preserved_in_done_info():
    callback = PercentMPCInjectCallback(
        domain="quadruped",
        task="velocity_tracking",
        target_percentage=25,
        verbose=0,
    )
    replay_buffer = _barrel_replay_buffer(n_envs=1)
    zero_obs = {
        "policy": np.zeros((1, 45)),
        "privileged": np.zeros((1, 4)),
    }
    for _ in range(210):
        replay_buffer.add(
            obs=zero_obs,
            next_obs=zero_obs,
            action=np.zeros((1, 12)),
            reward=np.zeros(1),
            done=np.zeros(1),
            infos=[{}],
            source=0,
        )
    captured_infos = []
    original_add = replay_buffer.add

    def _capture_add(*args, **kwargs):
        captured_infos.extend(kwargs["infos"])
        return original_add(*args, **kwargs)

    replay_buffer.add = _capture_add
    callback.init_callback(_DummyCallbackModel(replay_buffer, num_envs=1))
    trajectory = _valid_arrays()
    trajectory["success"] = np.asarray(False)
    trajectory["failure_reason"] = np.asarray("incomplete_roll")
    trajectory["classifier_result"][-1] = False
    callback._inject_saved_quadruped_transitions(trajectory)

    assert replay_buffer.dones[279, 0] == 1.0
    assert captured_infos[-1]["is_success"] is False
    assert captured_infos[-1]["failure_reason"] == "incomplete_roll"
    assert captured_infos[-1]["TimeLimit.truncated"] is False
