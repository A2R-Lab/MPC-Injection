"""Regression tests for quadruped model architectures built via train.py.

Run:
    conda run -n mpc-rl python -m pytest tests/test_quadruped_model_architecture.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch as th
from absl import flags
from torch import nn
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Ensure local imports resolve when pytest runs from the repo root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from mpc_rl.envs.domain_randomization import DomainRandomizationConfig
from mpc_rl.envs.go2_sysid import assert_go2_sysid_joint_dynamics
from mpc_rl.common import QuadrupedTensorboardCallback
import mpc_rl.train as train_module
from mpc_rl.train import (
    BARREL_ROLL_EVAL_SEEDS,
    BARREL_ROLL_FINAL_TEST_SEEDS,
    AllConfig,
    BarrelRollEvalCallback,
    BarrelRollPilotDiagnosticsCallback,
    barrel_roll_config_snapshot,
    barrel_roll_run_provenance,
    create_callbacks,
    create_model,
    evaluate_barrel_roll_policy,
    make_quadruped_env,
    quadruped_video_filename,
    validate_barrel_roll_training_options,
)

FLAGS = flags.FLAGS


if not FLAGS.is_parsed():
    FLAGS(["test_quadruped_model_architecture"])


def _build_cfg(algorithm: str) -> AllConfig:
    return AllConfig(
        algorithm=algorithm,
        learning_rate=3e-4,
        buffer_size=1_000,
        learning_starts=10,
        batch_size=32,
        tau=0.005,
        gamma=0.99,
        gradient_steps=1,
        policy_delay=2,
        seed=1,
        tensorboard_log="",
        inject_n_timesteps=5_000,
        inject_type="percentage",
        percentage=25,
        num_traj=10,
        random_select=True,
        data_dir="",
        quadruped_mpc_replay_mode="direct",
        use_go2_sysid=True,
        cheetah3_speed_goal=3.0,
    )


@pytest.fixture
def quadruped_vec_env():
    env = DummyVecEnv([
        lambda: make_quadruped_env(
            robot="go2",
            domain_rand_cfg=DomainRandomizationConfig.disabled(),
            simple_reward=False,
        )
    ])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    yield env
    env.close()


@pytest.fixture
def barrel_roll_vec_env():
    env = DummyVecEnv([
        lambda: make_quadruped_env(
            task="barrel_roll",
            robot="go2",
            domain_rand_cfg=DomainRandomizationConfig.disabled(),
            use_go2_sysid=True,
        )
    ])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    yield env
    env.close()


def _linear_layers(module: nn.Module) -> list[nn.Linear]:
    return [layer for layer in module.modules() if isinstance(layer, nn.Linear)]


def test_quadruped_sac_mpc_keeps_default_hidden_layers(quadruped_vec_env):
    model = create_model(quadruped_vec_env, _build_cfg("SAC-MPC"), is_quadruped=True)

    assert model.policy_kwargs.get("net_arch") is None

    actor_layers = _linear_layers(model.policy.actor.latent_pi)
    assert [layer.in_features for layer in actor_layers] == [45, 256]
    assert [layer.out_features for layer in actor_layers] == [256, 256]
    assert model.policy.actor.mu.in_features == 256
    assert model.policy.actor.mu.out_features == 12

    critic_q0_layers = _linear_layers(model.policy.critic.qf0)
    assert [layer.in_features for layer in critic_q0_layers] == [60, 256, 256]
    assert [layer.out_features for layer in critic_q0_layers] == [256, 256, 1]


def test_quadruped_sac_keeps_default_hidden_layers(quadruped_vec_env):
    model = create_model(quadruped_vec_env, _build_cfg("SAC"), is_quadruped=True)

    assert model.policy_kwargs.get("net_arch") is None

    actor_layers = _linear_layers(model.policy.actor.latent_pi)
    assert [layer.in_features for layer in actor_layers] == [45, 256]
    assert [layer.out_features for layer in actor_layers] == [256, 256]
    assert model.policy.actor.mu.in_features == 256
    assert model.policy.actor.mu.out_features == 12


def test_barrel_roll_sac_mpc_actor_critic_and_replay_dimensions(barrel_roll_vec_env):
    model = create_model(barrel_roll_vec_env, _build_cfg("SAC-MPC"), is_quadruped=True)

    assert model.replay_buffer_class.__name__ == "TaggedDictReplayBuffer"
    assert model.observation_space["policy"].shape == (45,)
    assert model.observation_space["privileged"].shape == (4,)
    assert model.action_space.shape == (12,)
    actor_layers = _linear_layers(model.policy.actor.latent_pi)
    assert actor_layers[0].in_features == 45
    critic_q0_layers = _linear_layers(model.policy.critic.qf0)
    assert critic_q0_layers[0].in_features == 61


def test_quadruped_td3_mpc_keeps_existing_architecture(quadruped_vec_env):
    model = create_model(quadruped_vec_env, _build_cfg("TD3-MPC"), is_quadruped=True)

    assert model.policy_kwargs.get("net_arch") is None

    actor_layers = _linear_layers(model.policy.actor.mu)
    assert [layer.in_features for layer in actor_layers] == [45, 400, 300]
    assert [layer.out_features for layer in actor_layers] == [400, 300, 12]


def test_make_quadruped_env_go2_uses_sysid_joint_dynamics():
    env = make_quadruped_env(
        robot="go2",
        domain_rand_cfg=DomainRandomizationConfig.disabled(),
        simple_reward=False,
    )
    try:
        assert_go2_sysid_joint_dynamics(env.unwrapped.mjModel)
    finally:
        env.close()


def test_make_quadruped_env_go2_can_disable_sysid_joint_dynamics():
    env = make_quadruped_env(
        robot="go2",
        domain_rand_cfg=DomainRandomizationConfig.disabled(),
        simple_reward=False,
        use_go2_sysid=False,
    )
    try:
        with pytest.raises(AssertionError, match="Go2 sysID joint dynamics"):
            assert_go2_sysid_joint_dynamics(env.unwrapped.mjModel)
    finally:
        env.close()


@pytest.mark.parametrize("algorithm", ["SAC", "TD3"])
def test_quadruped_pure_off_policy_uses_tensorboard_callback(tmp_path, algorithm):
    callbacks, eval_env, inject_callback = create_callbacks(
        cfg=_build_cfg(algorithm),
        enable_logging=False,
        logdir=tmp_path,
        domain="quadruped",
        task="velocity_tracking",
        seed=1,
        checkpoint_freq=25_000,
        eval_freq=10_000,
        num_envs=1,
        is_quadruped=True,
        robot="go2",
        save_replay_buffer_checkpoints=False,
        simple_reward=False,
        use_go2_sysid=True,
        domain_rand_config_type="disabled",
        cheetah3_speed_goal=3.0,
    )

    assert any(isinstance(callback, QuadrupedTensorboardCallback) for callback in callbacks)
    assert eval_env is None
    assert inject_callback is None


def test_quadruped_factory_routes_barrel_and_rejects_unknown_tasks():
    env = make_quadruped_env(
        task="barrel_roll",
        robot="go2",
        domain_rand_cfg=DomainRandomizationConfig.disabled(),
        use_go2_sysid=True,
    )
    try:
        assert env.spec.id == "QuadrupedBarrelRoll-v0"
        assert env.spec.max_episode_steps == 125
        assert env.observation_space["policy"].shape == (45,)
        assert env.observation_space["privileged"].shape == (4,)
        assert env.unwrapped.action_scale == 2.0
    finally:
        env.close()

    with pytest.raises(ValueError, match="Unknown quadruped task"):
        make_quadruped_env(task="unknown_task")
    with pytest.raises(ValueError, match="task-specific reward"):
        make_quadruped_env(task="barrel_roll", simple_reward=True)


def test_barrel_roll_callbacks_use_fixed_seed_success_evaluator(tmp_path):
    callbacks, eval_env, inject_callback = create_callbacks(
        cfg=_build_cfg("SAC"),
        enable_logging=True,
        logdir=tmp_path,
        domain="quadruped",
        task="barrel_roll",
        seed=1,
        checkpoint_freq=25_000,
        eval_freq=10_000,
        num_envs=4,
        is_quadruped=True,
        robot="go2",
        save_replay_buffer_checkpoints=False,
        simple_reward=False,
        use_go2_sysid=True,
        domain_rand_config_type="disabled",
        cheetah3_speed_goal=3.0,
    )
    try:
        evaluator = next(
            callback for callback in callbacks
            if isinstance(callback, BarrelRollEvalCallback)
        )
        assert evaluator.seeds == BARREL_ROLL_EVAL_SEEDS
        assert any(
            isinstance(callback, BarrelRollPilotDiagnosticsCallback)
            for callback in callbacks
        )
        assert inject_callback is None
    finally:
        eval_env.close()


@pytest.mark.parametrize(
    "override, message",
    [
        ({"robot": "aliengo"}, "robot"),
        ({"algorithm": "TD3-MPC"}, "algorithm"),
        ({"inject_type": "fixed"}, "inject_type"),
        ({"percentage": 20}, "percentage"),
        ({"replay_mode": "torque_saved_pd"}, "direct"),
        ({"domain_rand_enabled": True}, "randomization"),
        ({"domain_rand_config_type": "custom"}, "randomization"),
        ({"use_go2_sysid": False}, "use_go2_sysid"),
        ({"data_dir": None}, "data_dir"),
        ({"training_seed": 3}, "training seed"),
        ({"total_timesteps": 499_999}, "total_timesteps"),
        ({"num_envs": 8}, "num_envs"),
        ({"max_episode_steps": 1000}, "max_episode_steps"),
        ({"eval_freq": 5_000}, "eval_freq"),
        ({"checkpoint_freq": 10_000}, "checkpoint_freq"),
        ({"enable_logging": False}, "enable_logging"),
        ({"save_replay_buffer_checkpoints": True}, "save_replay_buffer_checkpoints"),
        ({"save_replay_buffer_final": True}, "save_replay_buffer_final"),
        ({"random_select": False}, "random_select"),
        ({"play_only": True}, "play_only"),
        ({"load_run_name": "resume"}, "load_run_name"),
        ({"checkpoint_evals": "10000"}, "checkpoint_evals"),
        ({"learning_rate": 1.0e-3}, "learning_rate"),
        ({"buffer_size": 100_000}, "buffer_size"),
        ({"learning_starts": 1_000}, "learning_starts"),
        ({"batch_size": 128}, "batch_size"),
        ({"tau": 0.01}, "tau"),
        ({"gamma": 0.95}, "gamma"),
        ({"gradient_steps": 1}, "gradient_steps"),
        ({"policy_delay": 1}, "policy_delay"),
    ],
)
def test_barrel_roll_training_rejects_incompatible_options(override, message):
    options = {
        "robot": "go2",
        "algorithm": "SAC-MPC",
        "inject_type": "percentage",
        "percentage": 25,
        "replay_mode": "direct",
        "domain_rand_enabled": False,
        "domain_rand_config_type": "disabled",
        "use_go2_sysid": True,
        "data_dir": "data/go2_barrel_roll/v4",
    }
    options.update(override)
    with pytest.raises(ValueError, match=message):
        validate_barrel_roll_training_options(**options)


def test_barrel_roll_training_rejects_zero_percent_injection_baseline():
    with pytest.raises(ValueError, match="percentage must be 25"):
        validate_barrel_roll_training_options(
            robot="go2",
            algorithm="SAC-MPC",
            inject_type="percentage",
            percentage=0,
            replay_mode="direct",
            domain_rand_enabled=False,
            domain_rand_config_type="disabled",
            use_go2_sysid=True,
            data_dir="data/go2_barrel_roll/v4",
        )


def test_barrel_roll_config_serializes_frozen_contract_and_held_out_seeds():
    snapshot = barrel_roll_config_snapshot("data/go2_barrel_roll/v4", 25)

    assert snapshot["task_id"] == "go2_barrel_roll"
    assert snapshot["schema_version"] == 4
    assert snapshot["robot"] == "go2"
    assert snapshot["roll_direction"] == 1.0
    assert snapshot["timing"]["control_steps"] == 125
    assert snapshot["timing"]["maneuver_horizon"] == 1.4
    assert snapshot["timing"]["episode_horizon"] == 2.5
    assert snapshot["action"]["scale"] == 2.0
    assert snapshot["action"]["lpf_cutoff_hz"] == 5.0
    assert len(snapshot["pd_kp"]) == 12
    assert snapshot["domain_randomization"] == "disabled"
    assert snapshot["go2_sysid_enabled"] is True
    assert snapshot["dataset"] == {
        "path": "data/go2_barrel_roll/v4",
        "replay_mode": "direct",
        "schema_version": 4,
        "target_mpc_percentage": 25,
    }
    assert snapshot["evaluation"]["checkpoint_selection_order"] == [
        "strict_success_rate_desc",
        "mean_final_hold_standing_score_desc",
        "timesteps_asc",
    ]
    assert snapshot["evaluation"]["steps"] == list(
        train_module.BARREL_ROLL_VALIDATION_STEPS
    )
    assert snapshot["evaluation"]["seeds"] == list(BARREL_ROLL_EVAL_SEEDS)
    assert snapshot["evaluation"]["final_test_seeds"] == list(
        BARREL_ROLL_FINAL_TEST_SEEDS
    )
    assert len(BARREL_ROLL_EVAL_SEEDS) == len(set(BARREL_ROLL_EVAL_SEEDS)) == 100
    assert BARREL_ROLL_EVAL_SEEDS == tuple(range(2_000_000, 2_000_100))
    assert BARREL_ROLL_FINAL_TEST_SEEDS == tuple(range(8_000_000, 8_000_100))
    assert set(BARREL_ROLL_EVAL_SEEDS).isdisjoint(BARREL_ROLL_FINAL_TEST_SEEDS)


def test_barrel_roll_run_provenance_verifies_dataset_hashes(monkeypatch, tmp_path):
    checksum_path = tmp_path / "checksums.sha256"
    aggregate_path = tmp_path / "generation_manifest_aggregate.jsonl"
    checksum_path.write_text("test checksum index\n", encoding="utf-8")
    aggregate_path.write_text('{"accepted": true}\n', encoding="utf-8")
    summary = {
        "schema_version": 4,
        "file_count": 2000,
        "transition_count": 250000,
        "effective_config_sha256": train_module.sha256_bytes(
            train_module.canonical_json(
                train_module.expected_effective_config()
            ).encode("utf-8")
        ),
        "checksum_index_sha256": train_module.sha256_file(checksum_path),
        "aggregate_manifest": aggregate_path.name,
        "aggregate_manifest_sha256": train_module.sha256_file(aggregate_path),
    }
    (tmp_path / "dataset_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    summary_hash = train_module.sha256_file(tmp_path / "dataset_summary.json")
    (tmp_path / "COMPLETE").write_text(
        json.dumps({
            "status": "complete",
            "file_count": 2000,
            "transition_count": 250000,
            "dataset_summary_sha256": summary_hash,
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        train_module,
        "_git_repository_snapshot",
        lambda path: {
            "commit": str(path.name).ljust(40, "0")[:40],
            "tracked_worktree_dirty": False,
        },
    )

    provenance = barrel_roll_run_provenance(str(tmp_path))

    assert provenance["dataset"]["file_count"] == 2000
    assert provenance["dataset"]["transition_count"] == 250000
    assert provenance["dataset"]["checksum_index_sha256"] == summary["checksum_index_sha256"]
    assert provenance["dataset"]["aggregate_manifest_sha256"] == summary["aggregate_manifest_sha256"]
    assert provenance["dataset"]["complete_marker_sha256"] == train_module.sha256_file(
        tmp_path / "COMPLETE"
    )
    assert set(provenance["runtime_source_sha256"]) == set(
        train_module.BARREL_ROLL_RUNTIME_SOURCE_PATHS
    )
    assert all(
        len(digest) == 64
        for digest in provenance["runtime_source_sha256"].values()
    )
    assert set(provenance["source"]) == {
        "root", "mpx", "primal_dual_ilqr", "gym_quadruped", "mujoco_mpc"
    }


def test_barrel_roll_run_provenance_requires_atomic_promotion_marker(tmp_path):
    (tmp_path / "dataset_summary.json").write_text("{}", encoding="utf-8")
    (tmp_path / "checksums.sha256").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="COMPLETE marker"):
        barrel_roll_run_provenance(str(tmp_path))


def test_barrel_roll_training_completion_requires_exact_artifacts(tmp_path):
    config = {
        "algorithm": "SAC-MPC",
        "env_name": "quadruped-barrel_roll",
        "seed": 1,
        "total_timesteps": 500_000,
        "num_envs": 4,
        "max_episode_steps": 125,
        "eval_freq": 10_000,
        "checkpoint_freq": 25_000,
        "inject_type": "percentage",
        "percentage": 25,
        "random_select": True,
        "quadruped_mpc_replay_mode": "direct",
        "use_go2_sysid": True,
        "save_replay_buffer_checkpoints": False,
        "save_replay_buffer_final": False,
        "barrel_roll": {
            "run_provenance": {
                "runtime_source_sha256": train_module._barrel_roll_runtime_source_hashes(
                    Path(train_module.__file__).resolve().parents[1]
                )
            }
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    for relative in (
        "final_model.zip",
        "vec_normalize.pkl",
        "best_model/best_model.zip",
        "best_model/vec_normalize.pkl",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    diagnostics_path = tmp_path / "barrel_roll_pilot_diagnostics.json"
    diagnostics_path.write_text(json.dumps({"status": "complete"}), encoding="utf-8")

    history = []
    for step in train_module.BARREL_ROLL_VALIDATION_STEPS:
        success_rate = 0.5 if step in (20_000, 30_000) else 0.1
        standing_score = 0.4 if step == 30_000 else 0.3
        history.append({
            "timesteps": step,
            "success_rate": success_rate,
            "mean_final_hold_standing_score": standing_score,
            "seeds": list(train_module.BARREL_ROLL_VALIDATION_SEEDS),
            "episodes": [{} for _ in train_module.BARREL_ROLL_VALIDATION_SEEDS],
            "failure_reasons": {},
            "final_hold_streak_distribution": {},
            "standing_subscore_distributions": {},
            "motion_distributions": {},
            "reward_component_distributions": {},
            "roll_progress_distributions": {},
            "contact_summary": {},
            "return_distribution": {},
        })
    (tmp_path / "barrel_roll_eval_history.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in history),
        encoding="utf-8",
    )
    selection = {
        "selection_order": [
            "strict_success_rate_desc",
            "mean_final_hold_standing_score_desc",
            "timesteps_asc",
        ],
        "success_rate": 0.5,
        "mean_final_hold_standing_score": 0.4,
        "timesteps": 30_000,
    }
    (tmp_path / "best_model/selection.json").write_text(
        json.dumps(selection), encoding="utf-8"
    )
    for step in range(25_000, 500_001, 25_000):
        (tmp_path / f"checkpoints/model_{step}_steps.zip").parent.mkdir(
            parents=True, exist_ok=True
        )
        (tmp_path / f"checkpoints/model_{step}_steps.zip").write_bytes(b"model")
        (tmp_path / f"checkpoints/model_vecnormalize_{step}_steps.pkl").write_bytes(
            b"stats"
        )
    event_path = tmp_path / "tensorboard/SAC_1/events.out.tfevents.test"
    event_path.parent.mkdir(parents=True)
    event_path.write_bytes(b"events")

    marker = train_module.write_barrel_roll_training_completion(
        tmp_path, SimpleNamespace(num_timesteps=500_000)
    )

    assert marker["status"] == "complete"
    assert marker["selected_timesteps"] == 30_000
    assert marker["replay_buffers_saved"] is False
    assert json.loads((tmp_path / "COMPLETE").read_text())["training_seed"] == 1


def test_barrel_roll_policy_evaluation_records_g9_episode_metrics():
    class _Policy:
        @staticmethod
        def obs_to_tensor(obs):
            return {
                key: th.as_tensor(value, dtype=th.float32)
                for key, value in obs.items()
            }, False

    class _Critic:
        @staticmethod
        def __call__(obs, actions):
            del obs
            batch = actions.shape[0]
            return th.ones((batch, 1)), th.full((batch, 1), 2.0)

    class _Model:
        policy = _Policy()
        critic = _Critic()
        device = th.device("cpu")

        @staticmethod
        def predict(obs, deterministic=True):
            del obs
            assert deterministic
            return np.zeros((1, 12), dtype=np.float32), None

    class _EvalEnv:
        def seed(self, seed):
            self.current_seed = seed

        def reset(self):
            self.step_count = 0
            return {"policy": np.zeros((1, 45)), "privileged": np.zeros((1, 4))}

        def step(self, action):
            assert action.shape == (1, 12)
            self.step_count += 1
            success = self.current_seed == 10
            done = self.step_count == 125
            contact_state = (
                np.zeros(4, dtype=bool)
                if self.step_count == 11
                else np.ones(4, dtype=bool)
            )
            stability_count = max(self.step_count - 10, 0)
            info = {
                "contact_state": contact_state,
                "raw_contact_state": contact_state,
                "stability_count": stability_count,
                "is_success": success if done else False,
                "failure_reason": None if success else "rotation_out_of_band",
                "roll_progress": 6.2 if success else 2.0,
                "roll_error": 0.08 if success else 4.28,
                "base_height": 0.27,
                "body_up_tilt": 0.1,
                "base_linear_speed": 0.05 if success else 0.2,
                "base_angular_speed": 0.3,
                "joint_velocity_norm": 0.4,
                "action_change_norm": 0.5,
                "final_hold_active": self.step_count > 100,
                "final_hold_streak": 25 if success and done else 0,
                "hold_conditions": {
                    "rotation": success,
                    "foot_support": True,
                    "height": True,
                    "tilt": True,
                    "base_linear_speed": success,
                    "base_angular_speed": True,
                    "joint_speed": True,
                },
                "final_hold_window_failure_counts": {
                    "rotation": 0 if success else 25,
                    "foot_support": 0,
                    "height": 0,
                    "tilt": 0,
                    "base_linear_speed": 0 if success else 25,
                    "base_angular_speed": 0,
                    "joint_speed": 0,
                },
                "hold_condition_failure_counts": {
                    "rotation": 0 if success else self.step_count,
                    "foot_support": 0,
                    "height": 0,
                    "tilt": 0,
                    "base_linear_speed": 0 if success else self.step_count,
                    "base_angular_speed": 0,
                    "joint_speed": 0,
                },
                "nonfoot_ground_contact_count": 0,
                "reward_components": {
                    "roll_tracking": 1.0,
                    "signed_progress": 0.25,
                    "standing_score": 0.6 if self.step_count >= 71 else 0.0,
                    "terminal_outcome": (25.0 if success else -25.0) if done else 0.0,
                    "foot_score": 1.0 if self.step_count >= 71 else 0.0,
                    "height_score": 0.9 if self.step_count >= 71 else 0.0,
                    "tilt_score": 0.8 if self.step_count >= 71 else 0.0,
                    "linear_speed_score": 0.7 if self.step_count >= 71 else 0.0,
                    "angular_speed_score": 0.6 if self.step_count >= 71 else 0.0,
                    "joint_speed_score": 0.5 if self.step_count >= 71 else 0.0,
                },
            }
            reward = sum(
                info["reward_components"][name]
                for name in (
                    "roll_tracking",
                    "signed_progress",
                    "standing_score",
                    "terminal_outcome",
                )
            )
            return {}, np.array([reward]), np.array([done]), [info]

    result = evaluate_barrel_roll_policy(
        _Model(), _EvalEnv(), seeds=(10, 11), include_critic=True
    )

    assert result["success_rate"] == pytest.approx(0.5)
    assert result["mean_return"] == pytest.approx(189.25)
    assert result["failure_reasons"]["rotation_out_of_band"] == 1
    assert result["failure_reasons"]["non_foot_ground_contact"] == 0
    assert result["touchdown_count"] == 2
    assert result["mean_touchdown_time_s"] == pytest.approx(0.24)
    assert result["stabilization_count"] == 2
    assert result["mean_stabilization_time_s"] == pytest.approx(0.30)
    assert [episode["seed"] for episode in result["episodes"]] == [10, 11]
    assert result["episodes"][0]["terminal_roll_error"] == pytest.approx(0.08)
    assert result["episodes"][0]["terminal_body_up_tilt"] == pytest.approx(0.1)
    assert result["episodes"][0]["final_hold_endpoint_count"] == 25
    assert result["mean_final_hold_standing_score"] == pytest.approx(0.6)
    assert result["mean_cumulative_reward_components"]["signed_progress"] == pytest.approx(
        31.25
    )
    assert result["critic_available"] is True
    assert result["critic_values"] == [
        {"min": 1.0, "max": 1.0, "mean": 1.0},
        {"min": 2.0, "max": 2.0, "mean": 2.0},
    ]


def test_selected_barrel_roll_checkpoint_writes_report_and_outcome_videos(
    monkeypatch, tmp_path
):
    best_dir = tmp_path / "best_model"
    best_dir.mkdir()
    (best_dir / "best_model.zip").touch()
    (best_dir / "vec_normalize.pkl").touch()
    (best_dir / "selection.json").write_text(
        json.dumps({"checkpoint_metric": "success_rate", "success_rate": 0.5, "timesteps": 50}),
        encoding="utf-8",
    )

    class _Env:
        closed = False

        def close(self):
            self.closed = True

    selected_env = _Env()
    selected_model = object()
    monkeypatch.setattr(
        train_module,
        "load_saved_model_for_video_eval",
        lambda **kwargs: (selected_model, selected_env),
    )
    monkeypatch.setattr(
        train_module,
        "evaluate_barrel_roll_policy",
        lambda *args, **kwargs: {
            "success_rate": 0.5,
            "episodes": [
                {"seed": 1_000_000, "success": True},
                {"seed": 1_000_001, "success": False},
            ],
        },
    )

    def _record(**kwargs):
        video_dir = kwargs["video_dir"]
        video_dir.mkdir(parents=True)
        labels = kwargs["barrel_roll_video_labels"]
        for seed, label in labels.items():
            (video_dir / f"{label}_seed{seed}.mp4").write_bytes(b"video")
        return {"episode_successes": [labels[seed] == "success" for seed in kwargs["barrel_roll_seeds"]]}

    monkeypatch.setattr(train_module, "evaluate_and_record", _record)

    report = train_module.evaluate_selected_barrel_roll_checkpoint(
        logdir=tmp_path,
        algorithm="SAC-MPC",
        domain="quadruped",
        task="barrel_roll",
        robot="go2",
        use_go2_sysid=True,
    )

    assert report["representative_video_seeds"] == {
        "success": 1_000_000,
        "failure": 1_000_001,
    }
    assert (best_dir / "evaluation.json").is_file()
    assert (best_dir / "videos" / "success_seed1000000.mp4").read_bytes() == b"video"
    assert (best_dir / "videos" / "failure_seed1000001.mp4").read_bytes() == b"video"
    assert selected_env.closed


def test_quadruped_video_names_are_task_aware():
    barrel_name = quadruped_video_filename(
        task="barrel_roll", episode=2, episode_seed=1_000_002, velocity=None
    )
    velocity_name = quadruped_video_filename(
        task="velocity_tracking", episode=2, episode_seed=2002, velocity=1.0
    )
    assert barrel_name == "rollout2_seed1000002.mp4"
    assert "vx" not in barrel_name
    assert velocity_name == "rollout2_vx1.0.mp4"


def test_barrel_roll_checkpoint_selection_uses_locked_order(monkeypatch, tmp_path):
    class _Logger:
        def record(self, *args, **kwargs):
            del args, kwargs

    class _Model:
        def __init__(self):
            self.logger = _Logger()
            self.saved = []
            self._env = SimpleNamespace(num_envs=1)

        def get_env(self):
            return self._env

        def save(self, path):
            self.saved.append(Path(path))

        def get_vec_normalize_env(self):
            return None

    results = iter(
        [
            {
                "success_rate": 0.50,
                "mean_final_hold_standing_score": 0.40,
                "mean_reward": 10.0,
                "failure_reasons": {},
            },
            {
                "success_rate": 0.50,
                "mean_final_hold_standing_score": 0.60,
                "mean_reward": 100.0,
                "failure_reasons": {},
            },
            {
                "success_rate": 0.50,
                "mean_final_hold_standing_score": 0.60,
                "mean_reward": 200.0,
                "failure_reasons": {},
            },
            {
                "success_rate": 0.60,
                "mean_final_hold_standing_score": 0.10,
                "mean_reward": 0.0,
                "failure_reasons": {},
            },
        ]
    )
    monkeypatch.setattr(train_module, "sync_envs_normalization", lambda *args: None)
    monkeypatch.setattr(
        train_module,
        "evaluate_barrel_roll_policy",
        lambda *args, **kwargs: next(results),
    )
    callback = BarrelRollEvalCallback(
        object(),
        eval_freq=1,
        best_model_save_path=tmp_path,
        seeds=BARREL_ROLL_EVAL_SEEDS,
        history_path=tmp_path / "history.jsonl",
    )
    model = _Model()
    callback.init_callback(model)
    for call in range(1, 5):
        callback.n_calls = call
        assert callback._on_step()

    assert model.saved == [tmp_path / "best_model"] * 3
    assert callback.best_success_rate == pytest.approx(0.60)
    assert callback.best_final_hold_standing_score == pytest.approx(0.10)
    selection = json.loads((tmp_path / "selection.json").read_text())
    assert selection == {
        "selection_order": [
            "strict_success_rate_desc",
            "mean_final_hold_standing_score_desc",
            "timesteps_asc",
        ],
        "success_rate": 0.60,
        "mean_final_hold_standing_score": 0.10,
        "timesteps": 0,
    }


def test_barrel_roll_evaluator_records_step_zero_history(monkeypatch, tmp_path):
    class _Logger:
        def __init__(self):
            self.records = {}

        def record(self, key, value, *args, **kwargs):
            del args, kwargs
            self.records[key] = value

    class _Model:
        num_timesteps = 0

        def __init__(self):
            self.logger = _Logger()
            self._env = SimpleNamespace(num_envs=1)

        def get_env(self):
            return self._env

        def save(self, path):
            Path(path).touch()

        def get_vec_normalize_env(self):
            return None

    monkeypatch.setattr(train_module, "sync_envs_normalization", lambda *args: None)
    monkeypatch.setattr(
        train_module,
        "evaluate_barrel_roll_policy",
        lambda *args, **kwargs: {
            "success_rate": 0.0,
            "mean_final_hold_standing_score": 0.0,
            "mean_reward": -1.0,
            "episode_rewards": [-1.0] * 100,
            "failure_reasons": {
                "non_foot_ground_contact:geom_29": 40,
                "non_foot_ground_contact:geom_53": 60,
            },
            "seeds": list(BARREL_ROLL_EVAL_SEEDS),
        },
    )
    history_path = tmp_path / "evaluation_history.jsonl"
    callback = BarrelRollEvalCallback(
        object(),
        eval_freq=1,
        best_model_save_path=tmp_path / "best",
        seeds=BARREL_ROLL_EVAL_SEEDS,
        history_path=history_path,
    )
    model = _Model()
    callback.init_callback(model)
    callback._on_training_start()

    records = [json.loads(line) for line in history_path.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["timesteps"] == 0
    assert records[0]["success_rate"] == 0.0
    assert records[0]["mean_final_hold_standing_score"] == 0.0
    assert records[0]["seeds"] == list(BARREL_ROLL_EVAL_SEEDS)
    assert model.logger.records["eval/failure_reason/non_foot_ground_contact"] == 100.0


def test_barrel_roll_pilot_diagnostics_proves_finite_signals(tmp_path):
    class _Logger:
        def __init__(self):
            self.name_to_value = {
                "train/actor_loss": 1.0,
                "train/critic_loss": 2.0,
                "train/ent_coef": 0.5,
                "train/ent_coef_loss": -0.25,
            }

        def record(self, key, value, *args, **kwargs):
            del args, kwargs
            self.name_to_value[key] = value

    class _Policy:
        @staticmethod
        def obs_to_tensor(observations):
            return {
                key: th.as_tensor(value, dtype=th.float32)
                for key, value in observations.items()
            }, False

    class _Critic:
        @staticmethod
        def __call__(observations, actions):
            batch = observations["policy"].shape[0]
            assert actions.shape == (batch, 12)
            return (th.ones((batch, 1)), th.full((batch, 1), 2.0))

    class _ReplayBuffer:
        full = False
        transition_sources = np.array([[0], [1]], dtype=np.uint8)

        @staticmethod
        def size():
            return 2

        @staticmethod
        def get_mpc_percentage():
            return 50.0

    class _Model:
        device = th.device("cpu")
        num_timesteps = 1
        _n_updates = 1
        policy = _Policy()
        critic = _Critic()
        replay_buffer = _ReplayBuffer()

        def __init__(self):
            self.logger = _Logger()
            self._env = SimpleNamespace(num_envs=1)

        def get_env(self):
            return self._env

    summary_path = tmp_path / "diagnostics.json"
    callback = BarrelRollPilotDiagnosticsCallback(summary_path)
    callback.init_callback(_Model())
    callback.locals = {
        "new_obs": {
            "policy": np.zeros((1, 45), dtype=np.float32),
            "privileged": np.zeros((1, 4), dtype=np.float32),
        },
        "actions": np.zeros((1, 12), dtype=np.float32),
        "rewards": np.array([1.0], dtype=np.float32),
        "infos": [{"roll_progress": 0.25}],
    }
    assert callback._on_step()
    callback._on_training_end()

    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "complete"
    assert summary["environment_transitions_checked"] == 1
    assert summary["q_batches_checked"] == 1
    assert summary["training_snapshots_checked"] == 1
    assert summary["ranges"]["q_value/0"] == [1.0, 1.0]
    assert summary["ranges"]["train/ent_coef"] == [0.5, 0.5]
    assert summary["replay_percentage_last"] == 50.0


def test_barrel_roll_pilot_diagnostics_fails_on_nonfinite_reward(tmp_path):
    callback = BarrelRollPilotDiagnosticsCallback(tmp_path / "diagnostics.json")
    callback._write_summary = lambda status, error=None: None
    callback.locals = {
        "new_obs": {"policy": np.zeros((1, 45)), "privileged": np.zeros((1, 4))},
        "actions": np.zeros((1, 12)),
        "rewards": np.array([np.nan]),
    }
    with pytest.raises(FloatingPointError, match="non-finite rewards"):
        callback._on_step()


def test_barrel_roll_tensorboard_logging_covers_task_and_replay_metrics():
    class _Logger:
        def __init__(self):
            self.records = {}

        def record(self, key, value, *args, **kwargs):
            del args, kwargs
            self.records[key] = value

    class _ReplayBuffer:
        @staticmethod
        def get_mpc_percentage():
            return 25.0

    class _Model:
        def __init__(self):
            self.logger = _Logger()
            self.replay_buffer = _ReplayBuffer()
            self._env = SimpleNamespace(num_envs=1)

        def get_env(self):
            return self._env

    callback = QuadrupedTensorboardCallback(log_freq=1, task="barrel_roll")
    model = _Model()
    callback.init_callback(model)
    callback.n_calls = 1
    callback.locals = {
        "actions": np.ones((1, 12)),
        "infos": [
            {
                "phase": 0.5,
                "roll_progress": 3.0,
                "roll_error": 0.1,
                "contact_state": np.array([True, False, True, False]),
                "stability_count": 2,
                "final_hold_streak": 0,
                "hold_valid": False,
                "hold_conditions": {
                    "rotation": True,
                    "foot_support": False,
                    "height": True,
                    "tilt": True,
                    "base_linear_speed": True,
                    "base_angular_speed": True,
                    "joint_speed": True,
                },
                "body_up_tilt": 0.2,
                "base_linear_speed": 0.05,
                "base_angular_speed": 0.25,
                "joint_velocity_norm": 0.5,
                "is_success": False,
                "failure_reason": "non_foot_ground_contact:geom_29",
                "action_clip_fraction": 0.25,
                "torque_saturation_fraction": 0.125,
                "applied_torques": np.zeros(12),
                "reward_components": {
                    "roll_tracking": 1.0,
                    "signed_progress": 0.25,
                    "standing_score": 0.75,
                    "foot_score": 0.5,
                    "height_score": 1.0,
                    "tilt_score": 0.9,
                    "linear_speed_score": 0.8,
                    "angular_speed_score": 0.7,
                    "joint_speed_score": 0.6,
                    "terminal_outcome": -25.0,
                },
            }
        ],
    }
    assert callback._on_step()

    expected_keys = {
        "barrel_roll/phase_mean",
        "barrel_roll/progress_mean",
        "barrel_roll/error_mean",
        "barrel_roll/contact_fraction",
        "barrel_roll/stability_count_mean",
        "barrel_roll/hold_valid_fraction",
        "barrel_roll/hold_condition/foot_support_valid_fraction",
        "barrel_roll/body_up_tilt_mean",
        "barrel_roll/base_linear_speed_mean",
        "barrel_roll/base_angular_speed_mean",
        "barrel_roll/joint_velocity_norm_mean",
        "barrel_roll/success_fraction",
        "barrel_roll/failure_reason/non_foot_ground_contact",
        "barrel_roll/action_clip_fraction",
        "barrel_roll/torque_saturation_fraction",
        "reward/roll_tracking",
        "reward/signed_progress",
        "reward/standing_score",
        "reward/foot_score",
        "reward/height_score",
        "reward/tilt_score",
        "reward/linear_speed_score",
        "reward/angular_speed_score",
        "reward/joint_speed_score",
        "reward/terminal_outcome",
        "replay_buffer/mpc_percentage_actual",
    }
    assert expected_keys <= model.logger.records.keys()


def test_barrel_roll_failure_metrics_aggregate_geom_details_without_collision():
    class _Logger:
        def __init__(self):
            self.records = {}

        def record(self, key, value, *args, **kwargs):
            del args, kwargs
            self.records[key] = value

    class _Model:
        def __init__(self):
            self.logger = _Logger()
            self._env = SimpleNamespace(num_envs=2)

        def get_env(self):
            return self._env

    callback = QuadrupedTensorboardCallback(log_freq=1, task="barrel_roll")
    model = _Model()
    callback.init_callback(model)
    callback.n_calls = 1
    callback.locals = {
        "infos": [
            {"failure_reason": "non_foot_ground_contact:geom_29"},
            {"failure_reason": "non_foot_ground_contact:geom_53"},
        ]
    }
    assert callback._on_step()
    assert model.logger.records == {
        "barrel_roll/failure_reason/non_foot_ground_contact": 2.0
    }
