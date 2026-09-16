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
    ACTION_SCALE,
    CONTROL_DT,
    CONTROL_STEPS,
    EPISODE_HORIZON,
    LEGACY_REWARD_CONFIG,
    REWARD_CONFIG,
    ROLL_DIRECTION_SIGN,
    ROLL_END_TIME,
    ROLL_START_TIME,
    SCHEMA_V2_VERSION,
    SCHEMA_V3_REWARD_CONFIG,
    SCHEMA_V3_VERSION,
    SCHEMA_VERSION,
    SIM_DT,
    body_up_tilt_from_quaternion,
    desired_roll_at_time,
    desired_roll_rate_at_time,
    final_hold_conditions,
    legacy_maneuver_phase_at_time,
    maneuver_phase_at_time,
    reward_config_dict,
    schema_v3_maneuver_phase_at_time,
    standing_subscores,
    success_config_dict,
)
from mpc_rl.planner.barrel_roll_dataset import (
    ACTION_SEMANTICS,
    BarrelRollValidationError,
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
    action_scale_for_schema,
    aggregate_barrel_roll_dataset,
    atomic_save_npz,
    canonical_json,
    expected_effective_config,
    dimensions_for_schema,
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


def _valid_arrays(
    seed: int = 0, schema_version: int = SCHEMA_VERSION
) -> dict[str, np.ndarray]:
    control_steps, physics_steps = dimensions_for_schema(schema_version)
    is_v3 = schema_version == SCHEMA_V3_VERSION
    is_v4 = schema_version == SCHEMA_VERSION
    phase_function = {
        1: legacy_maneuver_phase_at_time,
        SCHEMA_V2_VERSION: legacy_maneuver_phase_at_time,
        SCHEMA_V3_VERSION: schema_v3_maneuver_phase_at_time,
        SCHEMA_VERSION: maneuver_phase_at_time,
    }[schema_version]
    physics_time = np.arange(physics_steps + 1, dtype=np.float64) * SIM_DT
    control_time = np.arange(1, control_steps + 1, dtype=np.float64) * CONTROL_DT
    roll = np.array([desired_roll_at_time(time_s) for time_s in physics_time])
    qpos = np.zeros((19, physics_steps + 1), dtype=np.float64)
    qpos[2] = 0.27
    qpos[3] = np.cos(roll / 2.0)
    qpos[4] = np.sin(roll / 2.0)
    default_joint_pos = np.tile(np.array([0.0, 0.9, -1.8]), 4)
    qpos[7:] = default_joint_pos[:, None]
    qvel = np.zeros((18, physics_steps + 1), dtype=np.float64)
    if is_v3:
        qvel[3] = np.array([
            desired_roll_rate_at_time(time_s) for time_s in physics_time
        ])

    policy_all = np.zeros((control_steps + 1, 45), dtype=np.float64)
    privileged_all = np.zeros((control_steps + 1, 4), dtype=np.float64)
    for index in range(control_steps + 1):
        time_s = index * CONTROL_DT
        state_index = index * 4
        desired = desired_roll_at_time(time_s)
        rotation = Rotation.from_quat(
            np.roll(qpos[3:7, state_index], -1)
        ).as_matrix()
        policy_all[index, :3] = qvel[3:6, state_index]
        policy_all[index, 3:6] = rotation.T @ np.array([0.0, 0.0, -1.0])
        policy_all[index, 6:9] = (
            phase_function(time_s),
            np.sin(desired),
            np.cos(desired),
        )
        policy_all[index, 9:21] = qpos[7:, state_index] - default_joint_pos
        policy_all[index, 21:33] = qvel[6:, state_index]
        privileged_all[index, :3] = rotation.T @ qvel[:3, state_index]
    privileged_all[:, 3] = roll[::4]

    foot_contacts = np.zeros((physics_steps, 4), dtype=bool)
    stable_streak = np.zeros(control_steps, dtype=np.int64)
    if is_v4:
        foot_contacts[-100:] = True
        stable_streak[-25:] = np.arange(1, 26)
    elif not is_v3:
        foot_contacts[-20:] = True
        stable_streak[-5:] = np.arange(1, 6)
    classifier = np.zeros(control_steps, dtype=bool)
    classifier[-1] = True

    desired_ctrl = np.array([desired_roll_at_time(time_s) for time_s in control_time])
    pre_roll = roll[:-1:4]
    post_roll = roll[4::4]
    if is_v3:
        roll_tracking = SCHEMA_V3_REWARD_CONFIG.tracking_weight * np.exp(
            -(
                (desired_ctrl - post_roll)
                / SCHEMA_V3_REWARD_CONFIG.tracking_sigma
            ) ** 2
        )
        desired_rates = np.array([desired_roll_rate_at_time(t) for t in control_time])
        rate_tracking = SCHEMA_V3_REWARD_CONFIG.rate_tracking_weight * np.exp(
            -(
                (qvel[3, 4::4] - desired_rates)
                / SCHEMA_V3_REWARD_CONFIG.rate_tracking_sigma
            ) ** 2
        )
        action_change = np.zeros(control_steps)
        terminal_outcome = np.zeros(control_steps)
        terminal_outcome[-1] = SCHEMA_V3_REWARD_CONFIG.success_bonus
        rewards = roll_tracking + rate_tracking + action_change + terminal_outcome
    elif is_v4:
        roll_tracking = REWARD_CONFIG.tracking_weight * np.exp(
            -((desired_ctrl - post_roll) / REWARD_CONFIG.tracking_sigma) ** 2
        )
        expected_step = 2.0 * np.pi * CONTROL_DT / (
            ROLL_END_TIME - ROLL_START_TIME
        )
        signed_progress = REWARD_CONFIG.progress_weight * np.clip(
            ROLL_DIRECTION_SIGN * (post_roll - pre_roll) / expected_step,
            -REWARD_CONFIG.progress_clip,
            REWARD_CONFIG.progress_clip,
        )
        signed_progress[control_time > ROLL_END_TIME] = 0.0
    else:
        expected_step = 2.0 * np.pi * CONTROL_DT / (
            ROLL_END_TIME - ROLL_START_TIME
        )
        rewards = LEGACY_REWARD_CONFIG.tracking_weight * np.exp(
            -((desired_ctrl - post_roll) / LEGACY_REWARD_CONFIG.tracking_sigma) ** 2
        )
        rewards += LEGACY_REWARD_CONFIG.progress_weight * np.clip(
            ROLL_DIRECTION_SIGN * (post_roll - pre_roll) / expected_step,
            -LEGACY_REWARD_CONFIG.progress_clip,
            LEGACY_REWARD_CONFIG.progress_clip,
        )
        rewards[-1] += LEGACY_REWARD_CONFIG.success_bonus

    raw_foot_contacts_ctrl = np.vstack((
        np.zeros((1, 4), dtype=bool),
        foot_contacts[3::4],
    ))
    filtered_foot_contacts_ctrl = np.logical_or(
        raw_foot_contacts_ctrl[1:], raw_foot_contacts_ctrl[:-1]
    )
    if is_v4:
        hold_conditions = {
            name: np.zeros(control_steps, dtype=bool)
            for name in (
                "rotation",
                "foot_support",
                "height",
                "tilt",
                "base_linear_speed",
                "base_angular_speed",
                "joint_speed",
            )
        }
        standing_scores = {
            name: np.zeros(control_steps, dtype=np.float64)
            for name in (
                "foot_score",
                "height_score",
                "tilt_score",
                "linear_speed_score",
                "angular_speed_score",
                "joint_speed_score",
            )
        }
        base_height_ctrl = qpos[2, 4::4]
        body_up_tilt_ctrl = np.asarray(
            [
                body_up_tilt_from_quaternion(qpos[3:7, state_index])
                for state_index in range(4, physics_steps + 1, 4)
            ],
            dtype=np.float64,
        )
        base_linear_speed_ctrl = np.linalg.norm(qvel[:3, 4::4], axis=0)
        base_angular_speed_ctrl = np.linalg.norm(qvel[3:6, 4::4], axis=0)
        joint_velocity_norm_ctrl = np.linalg.norm(qvel[6:, 4::4], axis=0)
        final_hold_streak = np.zeros(control_steps, dtype=np.int64)
        streak = 0
        for index in range(control_steps):
            metrics = {
                "base_height": base_height_ctrl[index],
                "body_up_tilt": body_up_tilt_ctrl[index],
                "base_linear_speed": base_linear_speed_ctrl[index],
                "base_angular_speed": base_angular_speed_ctrl[index],
                "joint_velocity_norm": joint_velocity_norm_ctrl[index],
            }
            conditions = final_hold_conditions(
                filtered_foot_contacts=filtered_foot_contacts_ctrl[index],
                roll_progress=post_roll[index],
                **metrics,
            )
            scores = standing_subscores(
                filtered_foot_contacts=filtered_foot_contacts_ctrl[index],
                **metrics,
            )
            for name, value in conditions.items():
                hold_conditions[name][index] = value
            for name, value in scores.items():
                standing_scores[name][index] = value
            streak = streak + 1 if all(conditions.values()) else 0
            final_hold_streak[index] = streak
        standing_score = np.mean(
            np.column_stack(tuple(standing_scores.values())), axis=1
        )
        standing_score[control_time < REWARD_CONFIG.standing_start_time] = 0.0
        terminal_outcome = np.zeros(control_steps, dtype=np.float64)
        terminal_outcome[-1] = REWARD_CONFIG.success_bonus
        rewards = roll_tracking + signed_progress + standing_score + terminal_outcome

    terminated = np.zeros(control_steps, dtype=bool)
    terminated[-1] = True
    zeros_tau = np.zeros((12, physics_steps), dtype=np.float64)
    action_scale = action_scale_for_schema(schema_version)
    effective_json = canonical_json(expected_effective_config(schema_version))
    final_roll, final_pitch, _ = Rotation.from_quat(
        np.roll(qpos[3:7, -1], -1)
    ).as_euler("xyz")
    arrays = {
        "policy_obs": policy_all[:-1],
        "next_policy_obs": policy_all[1:],
        "privileged_obs": privileged_all[:-1],
        "next_privileged_obs": privileged_all[1:],
        "actions": np.zeros((control_steps, 12), dtype=np.float64),
        "rewards": rewards,
        "terminated_ctrl": terminated,
        "truncated_ctrl": np.zeros(control_steps, dtype=bool),
        "qpos": qpos,
        "qvel": qvel,
        "tau_applied": zeros_tau.copy(),
        "tau_mpx": zeros_tau.copy(),
        "tau_raw": zeros_tau.copy(),
        "q_des": zeros_tau.copy(),
        "dq_des": zeros_tau.copy(),
        "X_updates": np.zeros(
            (control_steps, MPC_NODES + 1, MPC_STATE_DIM), dtype=np.float32
        ),
        "U_updates": np.zeros((control_steps, MPC_NODES, 12), dtype=np.float32),
        "physics_time": physics_time,
        "control_time": control_time,
        "phase_ctrl": np.array([phase_function(time_s) for time_s in control_time]),
        "desired_roll_ctrl": desired_ctrl,
        "measured_roll_physics": roll,
        "foot_contacts": foot_contacts,
        "nonfoot_contact": np.full(physics_steps, "", dtype="<U96"),
        "stable_contact_streak": stable_streak,
        "classifier_result": classifier,
        "residual_actions_unclipped": np.zeros((control_steps, 12)),
        "action_clipped": np.zeros((control_steps, 12), dtype=bool),
        "mpx_saturation_by_actuator": np.zeros((12, physics_steps), dtype=bool),
        "applied_saturation_by_actuator": np.zeros((12, physics_steps), dtype=bool),
        "solve_seconds": np.full(control_steps, 0.01),
        "solve_objective_norm_sq": np.ones(control_steps),
        "solve_constraint_norm_sq": np.ones(control_steps),
        "solve_finite": np.ones(control_steps, dtype=bool),
        "default_joint_pos": default_joint_pos,
        "pd_kp": PD_KP.copy(),
        "pd_kd": PD_KD.copy(),
        "tracking_kp": TRACKING_KP.copy(),
        "tracking_kd": TRACKING_KD.copy(),
        "torque_limits": TORQUE_LIMITS.copy(),
        "schema_version": np.asarray(schema_version),
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
        "controller_steps": np.asarray(control_steps),
        "physics_steps": np.asarray(physics_steps),
        "maneuver_horizon": np.asarray(1.4),
        "mpc_dt": np.asarray(MPC_DT),
        "replanning_frequency_hz": np.asarray(REPLANNING_FREQUENCY_HZ),
        "action_scale": np.asarray(action_scale),
        "action_lpf_cutoff_hz": np.asarray(5.0),
        "action_lpf_alpha": np.asarray(1.0 - np.exp(-2.0 * np.pi * 5.0 * CONTROL_DT)),
        "action_semantics": np.asarray(ACTION_SEMANTICS),
        "saved_action_reproduces_lpf_transition": np.asarray(False),
        "contact_convention": np.asarray(CONTACT_CONVENTION),
        "controller_mode": np.asarray(CONTROLLER_MODE),
        "warm_start_policy": np.asarray(WARM_START_POLICY),
        "schedule_json": np.asarray(canonical_json(schedule_dict(schema_version))),
        "success_config_json": np.asarray(canonical_json(success_config_dict(schema_version))),
        "reward_config_json": np.asarray(canonical_json(reward_config_dict(schema_version))),
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
    phase_indices = np.minimum(np.arange(control_steps, dtype=np.int64) * 2, MPC_NODES)
    solve_replanned = np.ones(control_steps, dtype=bool)
    if is_v3 or is_v4:
        solve_replanned[int(round(1.4 / CONTROL_DT)):] = False
    solve_iterations = np.where(solve_replanned, 1, 0).astype(np.int64)
    solve_iteration_limits = solve_iterations.copy()
    solve_iterations[0] = 100
    solve_iteration_limits[0] = 100
    arrays.update(
        solve_iterations=solve_iterations,
        solve_iteration_limit=solve_iteration_limits,
        solve_replanned=solve_replanned,
        solve_phase_index=phase_indices,
        solve_elapsed_time=phase_indices.astype(np.float64) * MPC_DT,
        warm_start_shift=np.diff(phase_indices, prepend=phase_indices[0]),
    )
    if is_v3:
        arrays.update(
            reward_roll_tracking=roll_tracking,
            reward_rate_tracking=rate_tracking,
            reward_action_change=action_change,
            reward_terminal_outcome=terminal_outcome,
            episode_horizon=np.asarray(EPISODE_HORIZON),
            final_tilt=np.asarray(0.0),
        )
    elif is_v4:
        arrays.update(
            raw_foot_contacts_ctrl=raw_foot_contacts_ctrl,
            filtered_foot_contacts_ctrl=filtered_foot_contacts_ctrl,
            final_hold_streak=final_hold_streak,
            base_height_ctrl=base_height_ctrl,
            body_up_tilt_ctrl=body_up_tilt_ctrl,
            base_linear_speed_ctrl=base_linear_speed_ctrl,
            base_angular_speed_ctrl=base_angular_speed_ctrl,
            joint_velocity_norm_ctrl=joint_velocity_norm_ctrl,
            reward_roll_tracking=roll_tracking,
            reward_signed_progress=signed_progress,
            reward_standing_score=standing_score,
            reward_terminal_outcome=terminal_outcome,
            episode_horizon=np.asarray(EPISODE_HORIZON),
            final_tilt=np.asarray(0.0),
            final_base_linear_speed=np.asarray(base_linear_speed_ctrl[-1]),
            final_base_angular_speed=np.asarray(base_angular_speed_ctrl[-1]),
            final_joint_velocity_norm=np.asarray(joint_velocity_norm_ctrl[-1]),
            terminal_final_hold_streak=np.asarray(final_hold_streak[-1]),
            nonfoot_contact_count=np.asarray(0),
        )
        arrays.update({
            f"hold_{name}_valid": values
            for name, values in hold_conditions.items()
        })
        arrays.update({
            f"reward_{name}": values
            for name, values in standing_scores.items()
        })
    return {key: np.asarray(value) for key, value in arrays.items()}


def _write_valid(
    path: Path, seed: int = 0, schema_version: int = SCHEMA_VERSION
) -> Path:
    np.savez_compressed(path, **_valid_arrays(seed, schema_version))
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


def test_valid_schema_v4_loads_without_pickle_and_reports_saturation(tmp_path):
    path = _write_valid(tmp_path / "go2_barrel_roll_v4_dir_pos_seed_000000_ep_125.npz")
    with np.load(path, allow_pickle=False) as data:
        assert data["policy_obs"].shape == (125, 45)
        assert all(data[key].dtype.kind != "O" for key in data.files)
    report = validate_barrel_roll_file(path)
    assert report.valid, report.errors
    assert report.metrics["action_clip_fraction"] == 0.0
    assert report.metrics["mpx_torque_saturation_fraction"] == 0.0
    assert report.metrics["torque_saturation_fraction"] == 0.0
    assert report.metrics["cumulative_standing_score"] > 0.0
    with np.load(path, allow_pickle=False) as data:
        assert int(data["schema_version"]) == SCHEMA_VERSION
        assert float(data["action_scale"]) == ACTION_SCALE


def test_historical_schema_v1_remains_strictly_validatable(tmp_path):
    path = _write_valid(
        tmp_path / "go2_barrel_roll_v1_dir_pos_seed_000000_ep_070.npz",
        schema_version=1,
    )
    report = validate_barrel_roll_file(path)
    assert report.valid, report.errors
    with np.load(path, allow_pickle=False) as data:
        assert float(data["action_scale"]) == 0.5


def test_retained_schema_v2_remains_strictly_validatable(tmp_path):
    path = _write_valid(
        tmp_path / "go2_barrel_roll_v2_dir_pos_seed_000000_ep_070.npz",
        schema_version=SCHEMA_V2_VERSION,
    )
    report = validate_barrel_roll_file(path)
    assert report.valid, report.errors
    with np.load(path, allow_pickle=False) as data:
        assert data["policy_obs"].shape == (70, 45)
        assert float(data["action_scale"]) == 2.0
        assert "episode_horizon" not in data


def test_historical_schema_v3_remains_strictly_validatable(tmp_path):
    path = _write_valid(
        tmp_path / "go2_barrel_roll_v3_dir_pos_seed_000000_ep_125.npz",
        schema_version=SCHEMA_V3_VERSION,
    )
    report = validate_barrel_roll_file(path)
    assert report.valid, report.errors
    with np.load(path, allow_pickle=False) as data:
        assert data["policy_obs"].shape == (125, 45)
        assert float(data["action_scale"]) == 2.0
        assert float(data["episode_horizon"]) == EPISODE_HORIZON


def test_atomic_save_leaves_only_valid_final_archive(tmp_path):
    path = tmp_path / "go2_barrel_roll_v4_dir_pos_seed_000000_ep_125.npz"
    atomic_save_npz(path, _valid_arrays())
    assert validate_barrel_roll_file(path).valid
    assert list(tmp_path.iterdir()) == [path]


def _wrong_schema(arrays):
    arrays["schema_version"] = np.asarray(999)


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


def _incomplete_rotation(arrays):
    arrays["qpos"][3:7] = np.array([[1.0], [0.0], [0.0], [0.0]])


def _fallen_low_height(arrays):
    arrays["qpos"][2, -1] = 0.15


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
        _incomplete_rotation,
        _fallen_low_height,
        _wrong_reward_config,
    ],
    ids=lambda mutation: mutation.__name__.removeprefix("_"),
)
def test_required_corruptions_are_rejected(tmp_path, mutation):
    arrays = _valid_arrays()
    mutation(arrays)
    path = tmp_path / "go2_barrel_roll_v4_dir_pos_seed_000000_ep_125.npz"
    np.savez_compressed(path, **arrays)
    report = validate_barrel_roll_file(path)
    assert not report.valid
    assert report.errors


def test_schema_v3_nonfoot_contact_is_diagnostic_only(tmp_path):
    arrays = _valid_arrays(schema_version=SCHEMA_V3_VERSION)
    arrays["nonfoot_contact"][100] = "torso"
    path = tmp_path / "go2_barrel_roll_v3_nonfoot_diagnostic.npz"
    np.savez_compressed(path, **arrays)
    report = validate_barrel_roll_file(path)
    assert report.valid, report.errors


def test_schema_v4_rejects_nonfoot_contact_and_bad_previous_endpoint_filter(tmp_path):
    arrays = _valid_arrays()
    arrays["nonfoot_contact"][100] = "torso"
    arrays["nonfoot_contact_count"] = np.asarray(1)
    path = tmp_path / "go2_barrel_roll_v4_dir_pos_seed_000000_ep_125.npz"
    np.savez_compressed(path, **arrays)
    report = validate_barrel_roll_file(path)
    assert not report.valid
    assert "non-foot ground contact recorded" in report.errors

    arrays = _valid_arrays()
    arrays["filtered_foot_contacts_ctrl"] = arrays[
        "filtered_foot_contacts_ctrl"
    ].copy()
    arrays["filtered_foot_contacts_ctrl"][-1, 0] = False
    np.savez_compressed(path, **arrays)
    report = validate_barrel_roll_file(path)
    assert not report.valid
    assert "filtered_foot_contacts_ctrl mismatch" in report.errors


def test_schema_v4_rejects_filename_seed_inconsistency_and_short_final_hold(tmp_path):
    wrong_name = tmp_path / "go2_barrel_roll_v4_dir_pos_seed_000001_ep_125.npz"
    np.savez_compressed(wrong_name, **_valid_arrays(seed=0))
    report = validate_barrel_roll_file(wrong_name)
    assert not report.valid
    assert any(error.startswith("filename mismatch") for error in report.errors)

    arrays = _valid_arrays()
    arrays["raw_foot_contacts_ctrl"][-2:, 0] = False
    arrays["foot_contacts"][-5, 0] = False
    arrays["foot_contacts"][-1, 0] = False
    path = tmp_path / "go2_barrel_roll_v4_dir_pos_seed_000000_ep_125.npz"
    np.savez_compressed(path, **arrays)
    report = validate_barrel_roll_file(path)
    assert not report.valid
    assert any(
        "final hold condition failed: foot_support" in error
        or "fewer than 25 consecutive" in error
        for error in report.errors
    )


def test_integrity_checker_delegates_barrel_files_and_preserves_generic_npz(tmp_path):
    _write_valid(tmp_path / "go2_barrel_roll_v4_dir_pos_seed_000000_ep_125.npz")
    corrupted = _valid_arrays(seed=1)
    corrupted["task_id"] = np.asarray("velocity_tracking")
    np.savez_compressed(
        tmp_path / "go2_barrel_roll_v4_dir_pos_seed_000001_ep_125.npz",
        **corrupted,
    )
    generic = tmp_path / "quadruped_velocity_data.npz"
    np.savez_compressed(generic, qpos=np.zeros((2, 3)))
    assert check_npz_file(generic) == (True, None)
    valid, invalid, failures = check_directory(tmp_path, verbose=False)
    assert (valid, invalid) == (2, 1)
    assert failures[0][0].name == "go2_barrel_roll_v4_dir_pos_seed_000001_ep_125.npz"


def test_integrity_cli_exits_nonzero_for_barrel_corruption(tmp_path):
    arrays = _valid_arrays()
    arrays["rewards"][0] += 1.0
    path = tmp_path / "go2_barrel_roll_v4_dir_pos_seed_000000_ep_125.npz"
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
            tmp_path / f"go2_barrel_roll_v4_dir_pos_seed_{seed:06d}_ep_125.npz",
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
    assert summary["transition_count"] == 250
    assert summary["seeds"] == [0, 1]
    checksum_lines = (tmp_path / "checksums.sha256").read_text().splitlines()
    assert checksum_lines == [f"{sha256_file(path)}  {path.name}" for path in paths]
    saved_summary = json.loads((tmp_path / "dataset_summary.json").read_text())
    assert saved_summary["checksum_index_sha256"] == sha256_file(
        tmp_path / "checksums.sha256"
    )
    assert len((tmp_path / "generation_manifest_aggregate.jsonl").read_text().splitlines()) == 2


def test_aggregate_rejects_mixed_schema_versions(tmp_path):
    _write_valid(
        tmp_path / "go2_barrel_roll_v1_dir_pos_seed_000000_ep_070.npz",
        seed=0,
        schema_version=1,
    )
    _write_valid(
        tmp_path / "go2_barrel_roll_v2_dir_pos_seed_000001_ep_070.npz",
        seed=1,
        schema_version=2,
    )
    with pytest.raises(BarrelRollValidationError, match="mixes schema versions"):
        aggregate_barrel_roll_dataset(tmp_path)


def test_barrel_injection_is_strict_sorted_direct_multi_env_and_exact_25_percent(
    tmp_path, monkeypatch
):
    selected = _write_valid(
        tmp_path / "go2_barrel_roll_v4_dir_pos_seed_000001_ep_125.npz",
        seed=1,
    )
    first = _write_valid(
        tmp_path / "go2_barrel_roll_v4_dir_pos_seed_000000_ep_125.npz",
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
        expected_quadruped_schema_version=SCHEMA_VERSION,
        quadruped_mpc_replay_mode="direct",
        verbose=0,
    )
    assert callback.available_files == [first, selected]

    replay_buffer = _barrel_replay_buffer(n_envs=5)
    zero_obs = {
        "policy": np.zeros((5, 45)),
        "privileged": np.zeros((5, 4)),
    }
    for _ in range(75):
        replay_buffer.add(
            obs=zero_obs,
            next_obs=zero_obs,
            action=np.zeros((5, 12)),
            reward=np.zeros(5),
            done=np.zeros(5),
            infos=[{} for _ in range(5)],
            source=0,
        )

    captured_infos = []
    original_add = replay_buffer.add

    def _capture_add(*args, **kwargs):
        captured_infos.extend(kwargs["infos"])
        return original_add(*args, **kwargs)

    monkeypatch.setattr(replay_buffer, "add", _capture_add)
    callback.init_callback(_DummyCallbackModel(replay_buffer, num_envs=5))
    monkeypatch.setattr(
        callback,
        "_replay_quadruped_trajectory",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("barrel data must not construct or use velocity torque replay")
        ),
    )
    callback._inject_mpc_trajectories()

    assert replay_buffer.get_mpc_percentage() == pytest.approx(25.0)
    assert replay_buffer.size() == 100
    assert np.all(replay_buffer.transition_sources[75:100] == 1)
    with np.load(selected, allow_pickle=False) as data:
        injected_policy = replay_buffer.observations["policy"][75:100].reshape(125, 45)
        injected_next_policy = replay_buffer.next_observations["policy"][75:100].reshape(125, 45)
        injected_privileged = replay_buffer.observations["privileged"][75:100].reshape(125, 4)
        injected_next_privileged = replay_buffer.next_observations["privileged"][75:100].reshape(125, 4)
        injected_actions = replay_buffer.actions[75:100].reshape(125, 12)
        injected_rewards = replay_buffer.rewards[75:100].reshape(125)
        injected_dones = replay_buffer.dones[75:100].reshape(125)
        np.testing.assert_array_equal(
            injected_policy, data["policy_obs"].astype(injected_policy.dtype)
        )
        np.testing.assert_array_equal(
            injected_next_policy,
            data["next_policy_obs"].astype(injected_next_policy.dtype),
        )
        np.testing.assert_array_equal(
            injected_privileged,
            data["privileged_obs"].astype(injected_privileged.dtype),
        )
        np.testing.assert_array_equal(
            injected_next_privileged,
            data["next_privileged_obs"].astype(injected_next_privileged.dtype),
        )
        np.testing.assert_array_equal(injected_actions, data["actions"].astype(np.float32))
        np.testing.assert_array_equal(injected_rewards, data["rewards"].astype(np.float32))
        np.testing.assert_array_equal(
            injected_dones, data["terminated_ctrl"].astype(np.float32)
        )
    assert captured_infos[-1]["is_success"] is True
    assert captured_infos[-1]["failure_reason"] is None
    assert captured_infos[-1]["TimeLimit.truncated"] is False
    assert replay_buffer.timeouts[99, 4] == 0.0


def test_barrel_injection_bootstraps_at_exact_learning_start(tmp_path):
    trajectory = _write_valid(
        tmp_path / "go2_barrel_roll_v4_dir_pos_seed_000000_ep_125.npz"
    )
    callback = PercentMPCInjectCallback(
        domain="quadruped",
        task="barrel_roll",
        target_percentage=25,
        data_dir=str(tmp_path),
        random_select=False,
        trajectory_files=[trajectory.name],
        expected_quadruped_task=TASK_ID,
        expected_quadruped_schema_version=SCHEMA_VERSION,
        quadruped_mpc_replay_mode="direct",
        verbose=0,
    )
    replay_buffer = _barrel_replay_buffer(n_envs=4, buffer_size=20_000)
    zero_obs = {
        "policy": np.zeros((4, 45)),
        "privileged": np.zeros((4, 4)),
    }
    for _ in range(2_500):
        replay_buffer.add(
            obs=zero_obs,
            next_obs=zero_obs,
            action=np.zeros((4, 12)),
            reward=np.zeros(4),
            done=np.zeros(4),
            infos=[{} for _ in range(4)],
            source=0,
        )

    model = _DummyCallbackModel(replay_buffer, num_envs=4)
    model.learning_starts = 10_000
    callback.init_callback(model)
    callback.num_timesteps = 9_996
    assert callback._on_step()
    assert replay_buffer.get_mpc_percentage() == 0.0

    callback.num_timesteps = 10_000
    assert callback._on_step()
    stats = replay_buffer.get_composition_stats()
    assert stats["rl_transitions"] == 10_000
    assert stats["mpc_transitions"] == 3_336
    assert stats["mpc_percentage"] == pytest.approx(25.0, abs=0.1)

    assert callback._on_step()
    assert replay_buffer.get_composition_stats() == stats


def test_barrel_injection_rejects_any_malformed_file_without_fallback(tmp_path):
    _write_valid(tmp_path / "go2_barrel_roll_v4_dir_pos_seed_000000_ep_125.npz")
    malformed = _valid_arrays(seed=2)
    malformed["actions"][0, 0] = 2.0
    np.savez_compressed(
        tmp_path / "go2_barrel_roll_v4_dir_pos_seed_000002_ep_125.npz",
        **malformed,
    )

    with pytest.raises(ValueError, match="invalid barrel-roll trajectory"):
        PercentMPCInjectCallback(
            domain="quadruped",
            task="barrel_roll",
            target_percentage=25,
            data_dir=str(tmp_path),
            expected_quadruped_task=TASK_ID,
            expected_quadruped_schema_version=SCHEMA_VERSION,
            quadruped_mpc_replay_mode="direct",
            verbose=0,
        )


@pytest.mark.parametrize("old_schema", [1, 2, 3])
def test_schema_v4_injection_rejects_historical_data(tmp_path, old_schema):
    episode_steps = 70 if old_schema in (1, 2) else 125
    _write_valid(
        tmp_path / (
            f"go2_barrel_roll_v{old_schema}_dir_pos_seed_000000_"
            f"ep_{episode_steps:03d}.npz"
        ),
        schema_version=old_schema,
    )
    with pytest.raises(
        ValueError,
        match=rf"schema_version mismatch: expected 4, got {old_schema}",
    ):
        PercentMPCInjectCallback(
            domain="quadruped",
            task="barrel_roll",
            target_percentage=25,
            data_dir=str(tmp_path),
            expected_quadruped_task=TASK_ID,
            expected_quadruped_schema_version=SCHEMA_VERSION,
            quadruped_mpc_replay_mode="direct",
            verbose=0,
        )


def test_barrel_task_rejects_explicit_torque_replay(tmp_path):
    _write_valid(tmp_path / "go2_barrel_roll_v4_dir_pos_seed_000000_ep_125.npz")
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
    for _ in range(375):
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
    trajectory["failure_reason"] = np.asarray("rotation_out_of_band")
    trajectory["classifier_result"][-1] = False
    callback._inject_saved_quadruped_transitions(trajectory)

    assert replay_buffer.dones[499, 0] == 1.0
    assert captured_infos[-1]["is_success"] is False
    assert captured_infos[-1]["failure_reason"] == "rotation_out_of_band"
    assert captured_infos[-1]["TimeLimit.truncated"] is False
