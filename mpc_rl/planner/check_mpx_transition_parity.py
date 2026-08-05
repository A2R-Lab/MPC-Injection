"""Measure saved-action parity for the canonical MPX trajectory generator."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

import mpx.config.config_go2 as config
import mpx.utils.mpc_wrapper as mpc_wrapper
from mpc_rl.envs.domain_randomization import (
    DomainRandomizationConfig,
    extract_startup_domain_rand_patch,
    resolve_startup_domain_rand_config,
)
from mpc_rl.envs.velocity_tracking_env import QuadrupedVelocityTrackingEnv
from mpc_rl.planner.gen_traj_data_mpx_dr import generate_trajectory


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOLERANCES = (
    REPO_ROOT
    / "docs"
    / "mpx_bound_milestone_reports"
    / "transition_parity_tolerances.json"
)
DEFAULT_REPORT = (
    REPO_ROOT
    / "docs"
    / "mpx_bound_milestone_reports"
    / "transition_parity_baseline_report.json"
)


def _git_commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _numeric_metrics(expected: np.ndarray, actual: np.ndarray) -> dict:
    expected = np.asarray(expected, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if expected.shape != actual.shape:
        return {
            "shape_match": False,
            "expected_shape": list(expected.shape),
            "actual_shape": list(actual.shape),
            "finite": False,
            "max_abs": None,
            "rms": None,
            "per_control_step_max_abs": [],
        }

    difference = actual - expected
    finite = bool(np.all(np.isfinite(expected)) and np.all(np.isfinite(actual)))
    if difference.size == 0 or not finite:
        max_abs = None
        rms = None
        per_step = []
    else:
        absolute = np.abs(difference)
        max_abs = float(np.max(absolute))
        rms = float(np.sqrt(np.mean(np.square(difference))))
        if difference.ndim == 0:
            per_step = [max_abs]
        elif difference.ndim == 1:
            per_step = absolute.astype(float).tolist()
        else:
            per_step = np.max(
                absolute.reshape(absolute.shape[0], -1), axis=1
            ).astype(float).tolist()

    return {
        "shape_match": True,
        "expected_shape": list(expected.shape),
        "actual_shape": list(actual.shape),
        "finite": finite,
        "max_abs": max_abs,
        "rms": rms,
        "per_control_step_max_abs": per_step,
    }


def _metric_passes(metrics: dict, tolerance: dict) -> bool:
    return bool(
        metrics["shape_match"]
        and metrics["finite"]
        and metrics["max_abs"] is not None
        and metrics["max_abs"] <= tolerance["max_abs"]
        and metrics["rms"] <= tolerance["rms"]
    )


def _deterministic_push_schedule(scenario: dict) -> dict[int, np.ndarray]:
    """Convert the declared JSON push schedule to environment arrays."""
    return {
        int(event["after_control_step_zero_based"]): np.asarray(
            event["delta_qvel_xyz_rpy"], dtype=np.float64
        )
        for event in scenario.get("deterministic_push_schedule", [])
    }


def _make_replay_env(
    domain_rand_config_type: str,
    deterministic_push_schedule: dict[int, np.ndarray],
) -> QuadrupedVelocityTrackingEnv:
    _, domain_rand_cfg = resolve_startup_domain_rand_config(
        domain_rand_config_type
    )
    domain_rand_cfg.obs_noise_level = 0.0
    domain_rand_cfg.encoder_bias_range = (0.0, 0.0)
    return QuadrupedVelocityTrackingEnv(
        robot="go2",
        scene="flat",
        domain_rand_cfg=domain_rand_cfg,
        apply_startup_domain_rand_on_init=False,
        simple_reward=True,
        use_go2_sysid=True,
        enable_substep_diagnostics=True,
        deterministic_push_schedule=deterministic_push_schedule,
    )


def _replay_saved_actions(
    trajectory: dict,
    *,
    domain_rand_config_type: str,
    deterministic_push_schedule: dict[int, np.ndarray],
    seed: int,
    control_steps: int,
) -> dict:
    env = _make_replay_env(
        domain_rand_config_type, deterministic_push_schedule
    )
    try:
        patch = extract_startup_domain_rand_patch(trajectory)
        env.apply_startup_domain_rand_bundle(patch)
        first_command = np.asarray(trajectory["commands_ctrl"][0])
        env.set_commands(
            vx=float(first_command[0]),
            vy=float(first_command[1]),
            wz=float(first_command[2]),
        )
        initial_obs, _ = env.reset(seed=seed)

        saved_initial_qpos = np.asarray(trajectory["qpos"][:, 0])
        saved_initial_qvel = np.asarray(trajectory["qvel"][:, 0])
        reset_qpos_metrics = _numeric_metrics(saved_initial_qpos, env.mjData.qpos)
        reset_qvel_metrics = _numeric_metrics(saved_initial_qvel, env.mjData.qvel)
        forced_initial_state = not (
            np.array_equal(saved_initial_qpos, env.mjData.qpos)
            and np.array_equal(saved_initial_qvel, env.mjData.qvel)
        )
        if forced_initial_state:
            env.mjData.qpos[:] = saved_initial_qpos
            env.mjData.qvel[:] = saved_initial_qvel
            env.mjData.ctrl[:] = 0.0
            env.mjData.qacc_warmstart[:] = 0.0
            mujoco.mj_forward(env.mjModel, env.mjData)
            initial_obs = env._get_obs()

        qpos = []
        qvel = []
        policy_obs = []
        privileged_obs = []
        rewards = []
        terminated = []
        truncated = []
        applied_torques = []
        torque_saturation_mask = []
        action_clipping_mask = []
        push_delta_qvel = []

        available_steps = int(np.asarray(trajectory["actions"]).shape[0])
        replay_steps = min(control_steps, available_steps)
        current_command = first_command.copy()
        for control_step in range(replay_steps):
            command = np.asarray(trajectory["commands_ctrl"][control_step])
            if not np.array_equal(command, current_command):
                env.set_commands(
                    vx=float(command[0]),
                    vy=float(command[1]),
                    wz=float(command[2]),
                )
                current_command = command.copy()

            obs, reward, term, trunc, info = env.step(
                np.asarray(trajectory["actions"][control_step])
            )
            diagnostics = info["substep_diagnostics"]
            qpos.append(env.mjData.qpos.copy())
            qvel.append(env.mjData.qvel.copy())
            policy_obs.append(obs["policy"].copy())
            privileged_obs.append(obs["privileged"].copy())
            rewards.append(float(reward))
            terminated.append(bool(term))
            truncated.append(bool(trunc))
            applied_torques.append(diagnostics["applied_torques"].copy())
            torque_saturation_mask.append(
                diagnostics["torque_saturation_mask"].copy()
            )
            action_clipping_mask.append(
                diagnostics["action_clipping_mask"].copy()
            )
            push_delta_qvel.append(diagnostics["push_delta_qvel"].copy())

        return {
            "initial_policy_obs": initial_obs["policy"].copy(),
            "initial_privileged_obs": initial_obs["privileged"].copy(),
            "reset_qpos_metrics": reset_qpos_metrics,
            "reset_qvel_metrics": reset_qvel_metrics,
            "forced_initial_state": forced_initial_state,
            "control_steps": replay_steps,
            "qpos": np.asarray(qpos, dtype=np.float64),
            "qvel": np.asarray(qvel, dtype=np.float64),
            "policy_observation": np.asarray(policy_obs, dtype=np.float64),
            "privileged_observation": np.asarray(
                privileged_obs, dtype=np.float64
            ),
            "reward": np.asarray(rewards, dtype=np.float64),
            "termination": np.asarray(terminated, dtype=bool),
            "truncation": np.asarray(truncated, dtype=bool),
            "applied_torque": np.asarray(applied_torques, dtype=np.float64),
            "torque_saturation_mask": np.asarray(
                torque_saturation_mask, dtype=bool
            ),
            "action_clipping_mask": np.asarray(action_clipping_mask, dtype=bool),
            "push_delta_qvel": np.asarray(push_delta_qvel, dtype=np.float64),
        }
    finally:
        env.close()


def _source_fields(trajectory: dict, control_steps: int) -> dict:
    return {
        "qpos": np.asarray(trajectory["next_qpos_ctrl"][:control_steps]),
        "qvel": np.asarray(trajectory["next_qvel_ctrl"][:control_steps]),
        "policy_observation": np.asarray(
            trajectory["next_policy_obs"][:control_steps]
        ),
        "privileged_observation": np.asarray(
            trajectory["next_privileged_obs"][:control_steps]
        ),
        "reward": np.asarray(trajectory["rewards"][:control_steps]),
        "termination": np.asarray(
            trajectory["terminated_ctrl"][:control_steps]
        ),
        "truncation": np.asarray(trajectory["truncated_ctrl"][:control_steps]),
        "applied_torque": np.asarray(
            trajectory["applied_torques_ctrl"][:control_steps]
        ),
        "torque_saturation_mask": np.asarray(
            trajectory["torque_saturation_mask"][:control_steps]
        ),
        "push_delta_qvel": np.asarray(
            trajectory["push_delta_qvel"][:control_steps]
        ),
    }


def _compare_horizon(
    trajectory: dict,
    replay: dict,
    tolerances: dict,
    requested_steps: int,
) -> dict:
    compared_steps = min(requested_steps, replay["control_steps"])
    source = _source_fields(trajectory, compared_steps)
    numeric_results = {}
    numeric_pass = True
    for field, field_tolerance in tolerances["numeric_fields"].items():
        metrics = _numeric_metrics(source[field], replay[field])
        passed = _metric_passes(metrics, field_tolerance)
        metrics["tolerance"] = field_tolerance
        metrics["passed"] = passed
        numeric_results[field] = metrics
        numeric_pass = numeric_pass and passed

    termination_mismatches = int(
        np.count_nonzero(source["termination"] != replay["termination"])
    )
    truncation_mismatches = int(
        np.count_nonzero(source["truncation"] != replay["truncation"])
    )
    saturation_mismatch_fraction = float(
        np.mean(
            source["torque_saturation_mask"]
            != replay["torque_saturation_mask"]
        )
    )
    replay_action_clipping_fraction = float(
        np.mean(replay["action_clipping_mask"])
    )
    source_push_events = np.any(
        np.abs(source["push_delta_qvel"]) > 0.0, axis=1
    )
    replay_push_events = np.any(
        np.abs(replay["push_delta_qvel"]) > 0.0, axis=1
    )
    push_event_mismatches = int(
        np.count_nonzero(source_push_events != replay_push_events)
    )

    exact = tolerances["exact_fields"]
    masks = tolerances["mask_tolerances"]
    discrete_pass = bool(
        termination_mismatches <= exact["termination_mismatch_count"]
        and truncation_mismatches <= exact["truncation_mismatch_count"]
        and push_event_mismatches <= exact["push_event_mismatch_count"]
        and saturation_mismatch_fraction
        <= masks["torque_saturation_mismatch_fraction"]
        and replay_action_clipping_fraction
        <= masks["replay_action_clipping_fraction"]
    )
    complete = compared_steps == requested_steps

    return {
        "requested_control_steps": requested_steps,
        "compared_control_steps": compared_steps,
        "complete": complete,
        "numeric_fields": numeric_results,
        "termination_mismatch_count": termination_mismatches,
        "truncation_mismatch_count": truncation_mismatches,
        "torque_saturation_mismatch_fraction": saturation_mismatch_fraction,
        "replay_action_clipping_fraction": replay_action_clipping_fraction,
        "source_push_event_steps": np.flatnonzero(source_push_events)
        .astype(int)
        .tolist(),
        "replay_push_event_steps": np.flatnonzero(replay_push_events)
        .astype(int)
        .tolist(),
        "push_event_mismatch_count": push_event_mismatches,
        "passed": bool(complete and numeric_pass and discrete_pass),
    }


def _measure_scenario(
    scenario: dict,
    tolerances: dict,
    controller: mpc_wrapper.MPCControllerWrapper,
    verbose: int,
) -> dict:
    seed = int(tolerances["rollout_seed"])
    requested_steps = int(tolerances["short_rollout_control_steps"])
    push_schedule = _deterministic_push_schedule(scenario)
    trajectory = generate_trajectory(
        seed=seed,
        domain_rand_config_type=scenario["domain_rand_config_type"],
        dr_seed_offset=int(tolerances["dr_seed_offset"]),
        mpc=controller,
        episode_length=requested_steps,
        verbose=verbose,
        render=False,
        use_go2_sysid=True,
        deterministic_push_schedule=push_schedule,
        action_conversion_mode=tolerances["conversion_mode_under_test"],
    )

    one_step_replay = _replay_saved_actions(
        trajectory,
        domain_rand_config_type=scenario["domain_rand_config_type"],
        deterministic_push_schedule=push_schedule,
        seed=seed,
        control_steps=int(tolerances["one_step_control_steps"]),
    )
    full_replay = _replay_saved_actions(
        trajectory,
        domain_rand_config_type=scenario["domain_rand_config_type"],
        deterministic_push_schedule=push_schedule,
        seed=seed,
        control_steps=requested_steps,
    )
    one_step = _compare_horizon(
        trajectory,
        one_step_replay,
        tolerances,
        int(tolerances["one_step_control_steps"]),
    )
    short_rollout = _compare_horizon(
        trajectory, full_replay, tolerances, requested_steps
    )

    actions = np.asarray(trajectory["actions"])
    clipping_mask = np.asarray(trajectory["action_clipping_mask"])
    clipping_magnitude = np.asarray(trajectory["action_clipping_magnitude"])
    conversion_limits = tolerances["generation_action_conversion_limits"]
    clipped_fraction = float(np.mean(clipping_mask)) if clipping_mask.size else 0.0
    max_clip_magnitude = (
        float(np.max(clipping_magnitude)) if clipping_magnitude.size else 0.0
    )
    action_out_of_range_count = int(
        np.count_nonzero((actions < -1.0) | (actions > 1.0))
    )
    conversion_pass = bool(
        clipped_fraction <= conversion_limits["clipped_element_fraction"]
        and max_clip_magnitude <= conversion_limits["max_clip_magnitude"]
        and action_out_of_range_count
        <= tolerances["exact_fields"]["action_out_of_range_count"]
    )
    source_push_events = np.flatnonzero(
        np.any(np.abs(np.asarray(trajectory["push_delta_qvel"])) > 0.0, axis=1)
    )
    push_requirement_pass = bool(
        not scenario.get("require_push_event", False) or source_push_events.size > 0
    )
    dr_applied_fields = [
        str(value) for value in np.asarray(trajectory["dr_applied_fields"]).tolist()
    ]
    nontrivial_dr_pass = bool(
        scenario["domain_rand_config_type"] == "disabled" or dr_applied_fields
    )
    source_complete = bool(
        int(trajectory["completed_control_steps"]) == requested_steps
        and not bool(trajectory["fell"])
    )

    return {
        "name": scenario["name"],
        "domain_rand_config_type": scenario["domain_rand_config_type"],
        "seed": seed,
        "dr_seed": int(trajectory["dr_seed"]),
        "actual_mpx_output": True,
        "source_completed_control_steps": int(
            trajectory["completed_control_steps"]
        ),
        "source_fell": bool(trajectory["fell"]),
        "source_failure_reason": str(trajectory["failure_reason"]),
        "source_complete": source_complete,
        "dr_applied_fields": dr_applied_fields,
        "nontrivial_dr_pass": nontrivial_dr_pass,
        "initial_state_reproduction": {
            "one_step_reset_qpos": one_step_replay["reset_qpos_metrics"],
            "one_step_reset_qvel": one_step_replay["reset_qvel_metrics"],
            "one_step_forced_saved_state": one_step_replay[
                "forced_initial_state"
            ],
            "full_reset_qpos": full_replay["reset_qpos_metrics"],
            "full_reset_qvel": full_replay["reset_qvel_metrics"],
            "full_forced_saved_state": full_replay["forced_initial_state"],
        },
        "generation_action_conversion": {
            "clipped_element_fraction": clipped_fraction,
            "max_clip_magnitude": max_clip_magnitude,
            "action_out_of_range_count": action_out_of_range_count,
            "limits": conversion_limits,
            "passed": conversion_pass,
        },
        "source_torque_saturation_fraction": float(
            np.mean(np.asarray(trajectory["torque_saturation_mask"]))
        ),
        "source_push_event_steps": source_push_events.astype(int).tolist(),
        "push_requirement_pass": push_requirement_pass,
        "one_step": one_step,
        "short_rollout": short_rollout,
        "passed": bool(
            source_complete
            and nontrivial_dr_pass
            and conversion_pass
            and push_requirement_pass
            and one_step["passed"]
            and short_rollout["passed"]
        ),
    }


def run_parity_measurement(
    tolerances_path: Path,
    *,
    verbose: int = 1,
) -> dict:
    tolerance_bytes = tolerances_path.read_bytes()
    tolerances = json.loads(tolerance_bytes)
    controller = mpc_wrapper.MPCControllerWrapper(
        config,
        use_go2_sysid=True,
    )
    controller.robot_height = config.robot_height

    scenarios = []
    for scenario in tolerances["scenarios"]:
        if verbose:
            print(f"Measuring transition parity: {scenario['name']}")
        scenarios.append(
            _measure_scenario(scenario, tolerances, controller, verbose)
        )

    passed = all(scenario["passed"] for scenario in scenarios)
    return {
        "schema_version": 1,
        "report_type": "mpx_transition_parity_baseline",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "declaration_id": tolerances["declaration_id"],
        "tolerances_sha256": hashlib.sha256(tolerance_bytes).hexdigest(),
        "conversion_mode": tolerances["conversion_mode_under_test"],
        "root_commit": _git_commit(REPO_ROOT),
        "mpx_commit": _git_commit(REPO_ROOT / "deps" / "mpx"),
        "primal_dual_ilqr_commit": _git_commit(
            REPO_ROOT / "deps" / "mpx" / "mpx" / "primal_dual_ilqr"
        ),
        "mpx_model_sha256": _sha256(
            REPO_ROOT / "deps" / "mpx" / "mpx" / "data" / "go2" / "go2_mjx.xml"
        ),
        "rollout_model_sha256": _sha256(
            REPO_ROOT
            / "deps"
            / "gym-quadruped"
            / "gym_quadruped"
            / "robot_model"
            / "go2"
            / "go2.xml"
        ),
        "qrot_roll_cost": float(np.asarray(config.Qrot)[0, 0]),
        "tolerances": tolerances,
        "scenarios": scenarios,
        "passed": passed,
        "decision": (
            "preserve_existing_transition_path"
            if passed
            and tolerances["conversion_mode_under_test"]
            == "inferred_action_direct_torque_v1"
            else (
                "use_env_step_transition_path"
                if passed
                else (
                    "evaluate_conditional_env_step_mapping"
                    if tolerances["conversion_mode_under_test"]
                    == "inferred_action_direct_torque_v1"
                    else "stop_and_decide_action_interface"
                )
            )
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure actual-MPX saved-action transition parity"
    )
    parser.add_argument(
        "--tolerances",
        type=Path,
        default=DEFAULT_TOLERANCES,
        help="Predeclared parity-tolerance JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT,
        help="Machine-readable output report",
    )
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2])
    args = parser.parse_args()

    report = run_parity_measurement(args.tolerances, verbose=args.verbose)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Parity report: {args.output}")
    print(f"Passed: {report['passed']}")
    print(f"Decision: {report['decision']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
