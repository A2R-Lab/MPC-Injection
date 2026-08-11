"""Commission direct Go2 barrel-roll transitions with 50 Hz MPC replanning."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from time import perf_counter
from pathlib import Path

# Four production workers share one GPU.  Avoid JAX's default large up-front
# reservation while leaving solver numerics and controller configuration intact.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import gym_quadruped
import mediapy as media
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import mpx.config.config_barrel_roll as config
import mpx.utils.mpc_wrapper as mpc_wrapper
from mpc_rl.envs.barrel_roll_common import (
    ACTION_SCALE,
    CONTROL_DT,
    CONTROL_STEPS,
    EPISODE_HORIZON,
    LEGACY_CONTROL_STEPS,
    MANEUVER_HORIZON,
    ROLL_DIRECTION_SIGN,
    SCHEMA_VERSION,
    SCHEMA_V2_VERSION,
    body_up_tilt_from_quaternion,
    desired_roll_at_time,
    find_nonfoot_ground_contact,
    maneuver_phase_at_time,
)
from mpc_rl.envs.barrel_roll_env import QuadrupedBarrelRollEnv
from mpc_rl.planner.barrel_roll_dataset import (
    atomic_save_npz,
    build_provenance_fields,
    validate_barrel_roll_file,
)


TRACKING_KP = np.asarray(config.BARREL_TRACKING_KP, dtype=np.float64)
TRACKING_KD = np.asarray(config.BARREL_TRACKING_KD, dtype=np.float64)


def _sanitize_metrics(metrics: dict) -> tuple[dict, list[str]]:
    """Keep rejected non-finite attempts serializable without accepting them."""
    sanitized = dict(metrics)
    non_finite_fields = []
    for key, value in metrics.items():
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            sanitized[key] = None
            non_finite_fields.append(key)
    if non_finite_fields:
        sanitized["accepted"] = False
        sanitized["classifier_result"] = False
        sanitized["failure_reason"] = (
            sanitized.get("failure_reason")
            or "non_finite_metrics:" + ",".join(non_finite_fields)
        )
        sanitized["non_finite_metric_fields"] = non_finite_fields
    return sanitized, non_finite_fields


def _inverse_pd_residual_action(
    env: QuadrupedBarrelRollEnv,
    tau_applied: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    q = env.mjData.qpos[7:].copy()
    dq = env.mjData.qvel[6:].copy()
    q_target = q + (tau_applied + env.kd * dq) / env.kp
    unclipped = (q_target - env.default_joint_pos) / env.action_scale
    return unclipped, np.clip(unclipped, -1.0, 1.0)


def _render_saved_control_endpoints(
    env: QuadrupedBarrelRollEnv,
    qpos: np.ndarray,
    qvel: np.ndarray,
    video_path: Path,
) -> int:
    """Render the exact 125 saved control-endpoint states of an accepted rollout."""
    frames = []
    for control_step in range(CONTROL_STEPS):
        state_index = (control_step + 1) * env.decimation
        env.mjData.qpos[:] = qpos[:, state_index]
        env.mjData.qvel[:] = qvel[:, state_index]
        env.mjData.qacc[:] = 0.0
        env.mjData.time = (control_step + 1) * env.control_dt
        mujoco.mj_forward(env.mjModel, env.mjData)
        frame = env.render()
        if frame is None:
            raise RuntimeError(
                f"saved rollout rendering returned no frame at step {control_step + 1}"
            )
        frames.append(frame)
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    media.write_video(str(video_path), frames, fps=50)
    return len(frames)


def generate_attempt(
    seed: int,
    mpc: mpc_wrapper.MPCControllerWrapper,
    *,
    render: bool = False,
    render_video_path: Path | None = None,
    render_accepted_video_path: Path | None = None,
    nominal_spread_zero: bool = False,
    verbose: int = 1,
    generator_command: str | None = None,
    generator_config: dict | None = None,
    retained_v2_prefix: Path | None = None,
):
    """Execute one seeded receding-horizon attempt in the RL task plant."""
    attempt_started = perf_counter()
    render_mode = (
        "rgb_array"
        if render_video_path is not None or render_accepted_video_path is not None
        else ("human" if render else None)
    )
    env = QuadrupedBarrelRollEnv(
        render_mode=render_mode,
        use_go2_sysid=True,
        action_scale=ACTION_SCALE,
    )
    try:
        retained_prefix = None
        if retained_v2_prefix is not None:
            retained_v2_prefix = Path(retained_v2_prefix)
            report = validate_barrel_roll_file(retained_v2_prefix)
            if not report.valid:
                raise ValueError(
                    "invalid retained schema-v2 prefix: " + "; ".join(report.errors)
                )
            with np.load(retained_v2_prefix, allow_pickle=False) as loaded:
                retained_prefix = {key: loaded[key] for key in loaded.files}
            if int(retained_prefix["schema_version"]) != SCHEMA_V2_VERSION:
                raise ValueError("retained prefix must use schema version 2")
            if int(retained_prefix["rollout_seed"]) != seed:
                raise ValueError("retained prefix rollout seed does not match attempt seed")
            if nominal_spread_zero:
                raise ValueError("retained prefix is incompatible with nominal_spread_zero")
        reset_options = {"spread": 0.0} if nominal_spread_zero else None
        obs, _ = env.reset(seed=seed, options=reset_options)
        initial_qpos = env.mjData.qpos.copy()
        initial_qvel = env.mjData.qvel.copy()
        mpc.reset(initial_qpos, initial_qvel)
        if render and render_video_path is None:
            env.render()
        rendered_frames = []

        transitions = {
            key: []
            for key in (
                "policy_obs",
                "next_policy_obs",
                "privileged_obs",
                "next_privileged_obs",
                "actions",
                "rewards",
                "terminated_ctrl",
                "truncated_ctrl",
                "reward_roll_tracking",
                "reward_signed_progress",
                "reward_standing_score",
                "reward_terminal_outcome",
                "reward_foot_score",
                "reward_height_score",
                "reward_tilt_score",
                "reward_linear_speed_score",
                "reward_angular_speed_score",
                "reward_joint_speed_score",
            )
        }
        qpos = np.zeros((env.mjModel.nq, CONTROL_STEPS * env.decimation + 1))
        qvel = np.zeros((env.mjModel.nv, CONTROL_STEPS * env.decimation + 1))
        tau = np.zeros((env.num_joints, CONTROL_STEPS * env.decimation))
        tau_mpx = np.zeros_like(tau)
        tau_raw = np.zeros_like(tau)
        q_des = np.zeros_like(tau)
        dq_des = np.zeros_like(tau)
        mpx_saturation = np.zeros_like(tau, dtype=bool)
        applied_saturation = np.zeros_like(tau, dtype=bool)
        measured_roll = np.zeros(CONTROL_STEPS * env.decimation + 1)
        contacts = np.zeros((CONTROL_STEPS * env.decimation, 4), dtype=bool)
        nonfoot_contacts = np.full(
            CONTROL_STEPS * env.decimation, "", dtype="<U96"
        )
        phase_indices = np.zeros(CONTROL_STEPS * env.decimation, dtype=np.int64)
        horizon_positions = np.zeros(CONTROL_STEPS * env.decimation)
        qpos[:, 0] = initial_qpos
        qvel[:, 0] = initial_qvel

        X_updates = []
        U_updates = []
        solve_diagnostics = []
        stable_contact_streak = []
        final_hold_streak = []
        classifier_results = []
        raw_foot_contacts_ctrl = [env._previous_barrel_raw_contacts.copy()]
        filtered_foot_contacts_ctrl = []
        hold_conditions_ctrl = {
            name: []
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
        hold_metrics_ctrl = {
            name: []
            for name in (
                "base_height",
                "body_up_tilt",
                "base_linear_speed",
                "base_angular_speed",
                "joint_velocity_norm",
            )
        }
        residual_actions_unclipped = []
        min_base_height = float(initial_qpos[2])
        torque_saturation_count = 0
        mpx_saturation_count = 0
        max_joint_tracking_error = 0.0
        applied_physics_steps = 0
        viewer_closed = False
        local_failure_reason = None
        first_nonfoot_contact = None
        first_nonfoot_time = None
        replanning_active = True
        replan_stop_time = None
        retained_tail_plan = None

        for control_step in range(CONTROL_STEPS):
            elapsed_time = control_step * CONTROL_DT
            if replanning_active and elapsed_time >= MANEUVER_HORIZON:
                replanning_active = False
                if replan_stop_time is None:
                    replan_stop_time = MANEUVER_HORIZON
            stop_replanning_after_update = bool(
                replanning_active
                and elapsed_time >= config.BARREL_REPLAN_EARLIEST_STOP_TIME
                and np.all(env._get_foot_contacts())
                and env._stability_count >= config.BARREL_REPLAN_STABLE_STEPS
            )
            transitions["policy_obs"].append(obs["policy"].copy())
            transitions["privileged_obs"].append(obs["privileged"].copy())

            if retained_prefix is not None and control_step < LEGACY_CONTROL_STEPS:
                physics_start = control_step * env.decimation
                plan = {
                    "tau": retained_prefix["tau_mpx"][
                        :, physics_start : physics_start + env.decimation : 2
                    ].T,
                    "q": retained_prefix["q_des"][
                        :, physics_start : physics_start + env.decimation : 2
                    ].T,
                    "dq": retained_prefix["dq_des"][
                        :, physics_start : physics_start + env.decimation : 2
                    ].T,
                    "X": retained_prefix["X_updates"][control_step],
                    "U": retained_prefix["U_updates"][control_step],
                    "diagnostics": {
                        "iterations": int(retained_prefix["solve_iterations"][control_step]),
                        "iteration_limit": int(
                            retained_prefix["solve_iteration_limit"][control_step]
                        ),
                        "replanned": bool(retained_prefix["solve_replanned"][control_step]),
                        "phase_index": int(retained_prefix["solve_phase_index"][control_step]),
                        "elapsed_time": float(
                            retained_prefix["solve_elapsed_time"][control_step]
                        ),
                        "warm_start_shift": int(
                            retained_prefix["warm_start_shift"][control_step]
                        ),
                        "final_objective_norm_sq": float(
                            retained_prefix["solve_objective_norm_sq"][control_step]
                        ),
                        "final_constraint_norm_sq": float(
                            retained_prefix["solve_constraint_norm_sq"][control_step]
                        ),
                        "finite": bool(retained_prefix["solve_finite"][control_step]),
                        "solve_seconds": float(
                            retained_prefix["solve_seconds"][control_step]
                        ),
                    },
                }
            elif retained_prefix is not None:
                if retained_tail_plan is None:
                    shift = 2

                    def shift_and_pad(array):
                        return np.concatenate(
                            (array[shift:], np.repeat(array[-1:], shift, axis=0)),
                            axis=0,
                        )

                    tail_x = shift_and_pad(retained_prefix["X_updates"][-1])
                    tail_u = shift_and_pad(retained_prefix["U_updates"][-1])
                    retained_tail_plan = {
                        "tau": tail_u[:2, : env.num_joints],
                        "q": tail_x[1:3, 7 : 7 + env.num_joints],
                        "dq": tail_x[
                            1:3,
                            13 + env.num_joints : 13 + 2 * env.num_joints,
                        ],
                        "X": tail_x,
                        "U": tail_u,
                    }
                plan = {
                    **retained_tail_plan,
                    "diagnostics": {
                        "iterations": 0,
                        "iteration_limit": 0,
                        "replanned": False,
                        "phase_index": config.N,
                        "elapsed_time": MANEUVER_HORIZON,
                        "warm_start_shift": (
                            2 if control_step == LEGACY_CONTROL_STEPS else 0
                        ),
                        "final_objective_norm_sq": float(
                            retained_prefix["solve_objective_norm_sq"][-1]
                        ),
                        "final_constraint_norm_sq": float(
                            retained_prefix["solve_constraint_norm_sq"][-1]
                        ),
                        "finite": True,
                        "solve_seconds": 0.0,
                    },
                }
            else:
                plan = mpc.run_barrel_roll(
                    env.mjData.qpos.copy(),
                    env.mjData.qvel.copy(),
                    elapsed_time=elapsed_time,
                    iterations=(
                        config.BARREL_INITIAL_SOLVE_ITERATIONS
                        if control_step == 0
                        else 1
                    ),
                    replan=replanning_active,
                )
            if stop_replanning_after_update:
                replanning_active = False
                replan_stop_time = elapsed_time
            X_updates.append(plan["X"])
            U_updates.append(plan["U"])
            solve_diagnostics.append(plan["diagnostics"])
            if not plan["diagnostics"]["finite"]:
                local_failure_reason = "non_finite_solver_output"
                break

            segment_nodes = plan["tau"].shape[0]
            if segment_nodes <= 0 or env.decimation % segment_nodes:
                raise ValueError(
                    "MPC control segment must divide the environment decimation"
                )
            physics_steps_per_node = env.decimation // segment_nodes

            first_feedforward = plan["tau"][0]
            first_desired = plan["q"][0]
            current_q = env.mjData.qpos[7:].copy()
            current_dq = env.mjData.qvel[6:].copy()
            first_unclipped_tau = (
                first_feedforward
                + TRACKING_KP * (first_desired - current_q)
                - TRACKING_KD * current_dq
            )
            first_applied_tau = np.clip(
                first_unclipped_tau,
                env.torque_limits[:, 0],
                env.torque_limits[:, 1],
            )
            action_unclipped, action = _inverse_pd_residual_action(
                env, first_applied_tau
            )
            residual_actions_unclipped.append(action_unclipped)
            env._prev_last_action = env._last_action.copy()
            env._last_action = action.copy()
            env._before_control_step()

            for substep in range(env.decimation):
                node = substep // physics_steps_per_node
                feedforward = plan["tau"][node]
                desired = plan["q"][node]
                current_q = env.mjData.qpos[7:].copy()
                current_dq = env.mjData.qvel[6:].copy()
                unclipped_tau = (
                    feedforward
                    + TRACKING_KP * (desired - current_q)
                    - TRACKING_KD * current_dq
                )
                applied_tau = np.clip(
                    unclipped_tau,
                    env.torque_limits[:, 0],
                    env.torque_limits[:, 1],
                )
                torque_saturation_count += int(
                    np.count_nonzero(~np.isclose(unclipped_tau, applied_tau))
                )
                applied_physics_steps += 1
                mpx_saturation_count += int(
                    np.count_nonzero(
                        (feedforward < env.torque_limits[:, 0])
                        | (feedforward > env.torque_limits[:, 1])
                    )
                )
                max_joint_tracking_error = max(
                    max_joint_tracking_error,
                    float(np.max(np.abs(desired - current_q))),
                )
                sim_step = control_step * env.decimation + substep
                tau[:, sim_step] = applied_tau
                tau_mpx[:, sim_step] = feedforward
                tau_raw[:, sim_step] = unclipped_tau
                q_des[:, sim_step] = desired
                dq_des[:, sim_step] = plan["dq"][node]
                mpx_saturation[:, sim_step] = (
                    (feedforward < env.torque_limits[:, 0])
                    | (feedforward > env.torque_limits[:, 1])
                )
                applied_saturation[:, sim_step] = ~np.isclose(
                    unclipped_tau, applied_tau
                )
                phase_indices[sim_step] = plan["diagnostics"]["phase_index"]
                horizon_positions[sim_step] = (
                    plan["diagnostics"]["phase_index"] / config.N
                )
                env._applied_torques = applied_tau.copy()
                env.mjData.ctrl[:] = applied_tau
                mujoco.mj_step(env.mjModel, env.mjData)
                env._after_physics_substep()
                qpos[:, sim_step + 1] = env.mjData.qpos
                qvel[:, sim_step + 1] = env.mjData.qvel
                measured_roll[sim_step + 1] = env._roll_progress
                contacts[sim_step] = env._get_foot_contacts()
                contact_name = find_nonfoot_ground_contact(env)
                if contact_name is not None:
                    nonfoot_contacts[sim_step] = contact_name
                    if first_nonfoot_contact is None:
                        first_nonfoot_contact = contact_name
                        first_nonfoot_time = float(env.mjData.time)
                min_base_height = min(min_base_height, float(env.mjData.qpos[2]))

                if render and render_video_path is None:
                    env.render()
                    if env.viewer is not None and not env.viewer.is_running():
                        viewer_closed = True
                        break
                if env._physics_failure_reason is not None:
                    break

            env._step_count += 1
            env._steps_since_command_resample += 1
            env._update_feet_air_time()
            joint_velocity = env.mjData.qvel[6:].copy()
            env._joint_acc = (joint_velocity - env._last_joint_vel) / env.control_dt
            env._last_joint_vel = joint_velocity
            terminated = env._check_termination()
            reward = env._compute_reward(action, terminated)
            next_obs = env._get_obs()
            info = env._get_info()
            env._swing_peak *= ~env._current_contacts
            stable_contact_streak.append(env._stability_count)
            final_hold_streak.append(env._final_hold_streak)
            classifier_results.append(env._terminal_success())
            raw_foot_contacts_ctrl.append(info["raw_contact_state"].copy())
            filtered_foot_contacts_ctrl.append(info["contact_state"].copy())
            for name in hold_conditions_ctrl:
                hold_conditions_ctrl[name].append(info["hold_conditions"][name])
            for name in hold_metrics_ctrl:
                hold_metrics_ctrl[name].append(info[name])

            transitions["actions"].append(action.copy())
            transitions["rewards"].append(reward)
            transitions["terminated_ctrl"].append(terminated)
            transitions["truncated_ctrl"].append(False)
            transitions["reward_roll_tracking"].append(
                env._reward_components["roll_tracking"]
            )
            transitions["reward_signed_progress"].append(
                env._reward_components["signed_progress"]
            )
            transitions["reward_standing_score"].append(
                env._reward_components["standing_score"]
            )
            transitions["reward_terminal_outcome"].append(
                env._reward_components["terminal_outcome"]
            )
            for name in (
                "foot_score",
                "height_score",
                "tilt_score",
                "linear_speed_score",
                "angular_speed_score",
                "joint_speed_score",
            ):
                transitions[f"reward_{name}"].append(
                    env._reward_components[name]
                )
            transitions["next_policy_obs"].append(next_obs["policy"].copy())
            transitions["next_privileged_obs"].append(next_obs["privileged"].copy())
            obs = next_obs
            if render_video_path is not None:
                frame = env.render()
                if frame is not None:
                    rendered_frames.append(frame)

            if verbose > 1:
                print(
                    f"[seed={seed} step={control_step:02d}] "
                    f"solve={plan['diagnostics']['solve_seconds']:.3f}s "
                    f"roll={env._roll_progress:.3f}"
                )
            if viewer_closed:
                local_failure_reason = "viewer_closed"
                break
            if terminated and not info["is_success"]:
                local_failure_reason = info["failure_reason"] or "terminated"
                break

        completed_steps = len(transitions["rewards"])
        success = bool(
            completed_steps == CONTROL_STEPS
            and env._failure_reason is None
            and env._terminal_success()
            and not viewer_closed
        )
        failure_reason = None if success else (
            local_failure_reason or env._failure_reason or "incomplete_roll"
        )
        solve_times = np.asarray(
            [item["solve_seconds"] for item in solve_diagnostics], dtype=np.float64
        )
        replan_times = np.asarray(
            [
                item["solve_seconds"]
                for item in solve_diagnostics
                if item["replanned"]
            ],
            dtype=np.float64,
        )
        residual_unclipped = np.asarray(residual_actions_unclipped, dtype=np.float64)
        action_clip_fraction = float(
            np.mean(np.abs(residual_unclipped) > 1.0)
        ) if residual_unclipped.size else 0.0
        torque_saturation_fraction = float(
            torque_saturation_count / (applied_physics_steps * env.num_joints)
        ) if applied_physics_steps else 0.0
        mpx_saturation_fraction = float(
            mpx_saturation_count / (applied_physics_steps * env.num_joints)
        ) if applied_physics_steps else 0.0
        constraint_norms = np.asarray(
            [item["final_constraint_norm_sq"] for item in solve_diagnostics],
            dtype=np.float64,
        )
        final_roll, final_pitch = env._upright_errors()
        final_tilt = body_up_tilt_from_quaternion(env.mjData.qpos[3:7])
        final_contacts = env._filtered_barrel_contacts.copy()
        reward_roll_tracking = np.asarray(
            transitions["reward_roll_tracking"], dtype=np.float64
        )
        reward_signed_progress = np.asarray(
            transitions["reward_signed_progress"], dtype=np.float64
        )
        reward_standing_score = np.asarray(
            transitions["reward_standing_score"], dtype=np.float64
        )
        metrics = {
            "seed": seed,
            "accepted": success,
            "failure_reason": failure_reason,
            "completed_control_steps": completed_steps,
            "roll_progress": float(env._roll_progress),
            "final_roll": final_roll,
            "final_pitch": final_pitch,
            "final_base_height": float(env.mjData.qpos[2]),
            "final_tilt": final_tilt,
            "final_base_linear_speed": float(np.linalg.norm(env.mjData.qvel[:3])),
            "final_base_angular_speed": float(np.linalg.norm(env.mjData.qvel[3:6])),
            "final_joint_velocity_norm": float(np.linalg.norm(env.mjData.qvel[6:])),
            "stable_contact_streak": int(env._stability_count),
            "final_hold_streak": int(env._final_hold_streak),
            "final_contacts": final_contacts.tolist(),
            "classifier_result": bool(env._terminal_success()),
            "nonfoot_contact_count": int(
                np.count_nonzero(nonfoot_contacts[:applied_physics_steps] != "")
            ),
            "first_nonfoot_contact": first_nonfoot_contact,
            "first_nonfoot_time": first_nonfoot_time,
            "sampled_spread": float(env._spread),
            "min_base_height": min_base_height,
            "mean_solve_seconds": float(np.mean(solve_times)) if solve_times.size else None,
            "mean_replan_seconds": (
                float(np.mean(replan_times[1:]))
                if replan_times.size > 1
                else None
            ),
            "max_solve_seconds": float(np.max(solve_times)) if solve_times.size else None,
            "initial_solve_iterations": (
                int(solve_diagnostics[0]["iterations"])
                if solve_diagnostics
                else 0
            ),
            "final_constraint_norm_sq": (
                float(constraint_norms[-1])
                if constraint_norms.size and np.isfinite(constraint_norms[-1])
                else None
            ),
            "max_constraint_norm_sq": (
                float(np.max(constraint_norms[np.isfinite(constraint_norms)]))
                if np.isfinite(constraint_norms).any()
                else None
            ),
            "final_objective_norm_sq": (
                float(solve_diagnostics[-1]["final_objective_norm_sq"])
                if solve_diagnostics
                else None
            ),
            "solver_finite": bool(
                solve_diagnostics
                and all(item["finite"] for item in solve_diagnostics)
            ),
            "action_clip_fraction": action_clip_fraction,
            "max_unclipped_action": (
                float(np.max(np.abs(residual_unclipped)))
                if residual_unclipped.size
                else 0.0
            ),
            "mpx_torque_saturation_fraction": mpx_saturation_fraction,
            "torque_saturation_fraction": torque_saturation_fraction,
            "max_joint_tracking_error": max_joint_tracking_error,
            "go2_sysid_enabled": True,
            "controller_mode": "phase_limited_replanning",
            "replan_earliest_stop_time": (
                config.BARREL_REPLAN_EARLIEST_STOP_TIME
            ),
            "replan_stop_time": replan_stop_time,
            "replan_stable_steps": config.BARREL_REPLAN_STABLE_STEPS,
            "replan_update_count": int(
                sum(item["replanned"] for item in solve_diagnostics)
            ),
            "maneuver_horizon": MANEUVER_HORIZON,
            "episode_horizon": EPISODE_HORIZON,
            "cumulative_roll_tracking": float(np.sum(reward_roll_tracking)),
            "cumulative_signed_progress": float(np.sum(reward_signed_progress)),
            "cumulative_standing_score": float(np.sum(reward_standing_score)),
            "attempt_seconds": float(perf_counter() - attempt_started),
        }
        if render_video_path is not None:
            metrics["rendered_frame_count"] = len(rendered_frames)
            metrics["video_complete"] = len(rendered_frames) == CONTROL_STEPS
        valid_state_count = applied_physics_steps + 1
        valid_control_count = completed_steps
        state_times = np.arange(valid_state_count, dtype=np.float64) * env.sim_dt
        state_quaternions = qpos[3:7, :valid_state_count].T
        state_euler = Rotation.from_quat(
            np.roll(state_quaternions, -1, axis=1)
        ).as_euler("xyz")
        trace = {
            "time": state_times,
            "desired_unwrapped_roll": np.asarray(
                [desired_roll_at_time(time_s) for time_s in state_times]
            ),
            "measured_unwrapped_roll": measured_roll[:valid_state_count],
            "base_position": qpos[:3, :valid_state_count].T,
            "base_quaternion_wxyz": state_quaternions,
            "base_euler_xyz": state_euler,
            "joint_position": qpos[7:, :valid_state_count].T,
            "joint_velocity": qvel[6:, :valid_state_count].T,
            "joint_position_desired": q_des[:, :applied_physics_steps].T,
            "joint_velocity_desired": dq_des[:, :applied_physics_steps].T,
            "joint_position_error": (
                q_des[:, :applied_physics_steps]
                - qpos[7:, :applied_physics_steps]
            ).T,
            "joint_velocity_error": (
                dq_des[:, :applied_physics_steps]
                - qvel[6:, :applied_physics_steps]
            ).T,
            "tau_mpx": tau_mpx[:, :applied_physics_steps].T,
            "tau_raw": tau_raw[:, :applied_physics_steps].T,
            "tau_applied": tau[:, :applied_physics_steps].T,
            "actuator_ctrlrange": env.torque_limits.copy(),
            "mpx_saturation_by_actuator": mpx_saturation[
                :, :applied_physics_steps
            ].T,
            "applied_saturation_by_actuator": applied_saturation[
                :, :applied_physics_steps
            ].T,
            "foot_contacts": contacts[:applied_physics_steps],
            "nonfoot_contact": nonfoot_contacts[:applied_physics_steps],
            "phase_index": phase_indices[:applied_physics_steps],
            "horizon_position": horizon_positions[:applied_physics_steps],
            "solve_objective_norm_sq": np.asarray(
                [item["final_objective_norm_sq"] for item in solve_diagnostics]
            ),
            "solve_constraint_norm_sq": constraint_norms,
            "solve_finite": np.asarray(
                [item["finite"] for item in solve_diagnostics], dtype=bool
            ),
            "solve_replanned": np.asarray(
                [item["replanned"] for item in solve_diagnostics], dtype=bool
            ),
            "warm_start_shift": np.asarray(
                [item["warm_start_shift"] for item in solve_diagnostics],
                dtype=np.int64,
            ),
            "stable_contact_streak": np.asarray(
                stable_contact_streak[:valid_control_count], dtype=np.int64
            ),
            "final_hold_streak": np.asarray(
                final_hold_streak[:valid_control_count], dtype=np.int64
            ),
            "raw_foot_contacts_ctrl": np.asarray(
                raw_foot_contacts_ctrl[: valid_control_count + 1], dtype=bool
            ),
            "filtered_foot_contacts_ctrl": np.asarray(
                filtered_foot_contacts_ctrl[:valid_control_count], dtype=bool
            ),
            **{
                f"hold_{name}_valid": np.asarray(
                    values[:valid_control_count], dtype=bool
                )
                for name, values in hold_conditions_ctrl.items()
            },
            **{
                f"{name}_ctrl": np.asarray(
                    values[:valid_control_count], dtype=np.float64
                )
                for name, values in hold_metrics_ctrl.items()
            },
            "classifier_result": np.asarray(
                classifier_results[:valid_control_count], dtype=bool
            ),
            "rollout_success": np.asarray(success),
            "sampled_spread": np.asarray(env._spread),
            "tracking_kp": TRACKING_KP.copy(),
            "tracking_kd": TRACKING_KD.copy(),
            "go2_sysid_enabled": np.asarray(True),
            "controller_mode": np.asarray("phase_limited_replanning"),
            "replan_earliest_stop_time": np.asarray(
                config.BARREL_REPLAN_EARLIEST_STOP_TIME
            ),
            "replan_stop_time": np.asarray(
                -1.0 if replan_stop_time is None else replan_stop_time
            ),
            "replan_stable_steps": np.asarray(
                config.BARREL_REPLAN_STABLE_STEPS
            ),
            "maneuver_horizon": np.asarray(MANEUVER_HORIZON),
            "episode_horizon": np.asarray(EPISODE_HORIZON),
        }
        if render_video_path is not None:
            if success and len(rendered_frames) != CONTROL_STEPS:
                raise RuntimeError(
                    "successful rollout did not render exactly "
                    f"{CONTROL_STEPS} frames: {len(rendered_frames)}"
                )
            if rendered_frames:
                render_video_path = Path(render_video_path)
                render_video_path.parent.mkdir(parents=True, exist_ok=True)
                media.write_video(str(render_video_path), rendered_frames, fps=50)
        if render_accepted_video_path is not None and success:
            rendered_count = _render_saved_control_endpoints(
                env, qpos, qvel, render_accepted_video_path
            )
            metrics["rendered_frame_count"] = rendered_count
            metrics["video_complete"] = rendered_count == CONTROL_STEPS
        if not success:
            return None, metrics, trace

        data = {key: np.asarray(value) for key, value in transitions.items()}
        data.update(
            qpos=qpos,
            qvel=qvel,
            tau_applied=tau,
            tau_mpx=tau_mpx,
            q_des=q_des,
            dq_des=dq_des,
            tau_raw=tau_raw,
            X_updates=np.asarray(X_updates),
            U_updates=np.asarray(U_updates),
            physics_time=np.arange(
                CONTROL_STEPS * env.decimation + 1, dtype=np.float64
            ) * env.sim_dt,
            control_time=np.arange(1, CONTROL_STEPS + 1, dtype=np.float64)
            * env.control_dt,
            phase_ctrl=np.asarray(
                [
                    maneuver_phase_at_time((step + 1) * env.control_dt)
                    for step in range(CONTROL_STEPS)
                ],
                dtype=np.float64,
            ),
            desired_roll_ctrl=np.asarray(
                [
                    desired_roll_at_time((step + 1) * env.control_dt)
                    for step in range(CONTROL_STEPS)
                ],
                dtype=np.float64,
            ),
            measured_roll_physics=measured_roll,
            foot_contacts=contacts,
            nonfoot_contact=nonfoot_contacts,
            stable_contact_streak=np.asarray(stable_contact_streak, dtype=np.int64),
            final_hold_streak=np.asarray(final_hold_streak, dtype=np.int64),
            raw_foot_contacts_ctrl=np.asarray(raw_foot_contacts_ctrl, dtype=bool),
            filtered_foot_contacts_ctrl=np.asarray(
                filtered_foot_contacts_ctrl, dtype=bool
            ),
            classifier_result=np.asarray(classifier_results, dtype=bool),
            action_clipped=np.abs(residual_unclipped) > 1.0,
            mpx_saturation_by_actuator=mpx_saturation,
            applied_saturation_by_actuator=applied_saturation,
            solve_seconds=solve_times,
            solve_iterations=np.asarray(
                [item["iterations"] for item in solve_diagnostics], dtype=np.int64
            ),
            solve_iteration_limit=np.asarray(
                [item["iteration_limit"] for item in solve_diagnostics],
                dtype=np.int64,
            ),
            solve_objective_norm_sq=np.asarray(
                [item["final_objective_norm_sq"] for item in solve_diagnostics]
            ),
            solve_constraint_norm_sq=np.asarray(
                [item["final_constraint_norm_sq"] for item in solve_diagnostics]
            ),
            solve_finite=np.asarray(
                [item["finite"] for item in solve_diagnostics], dtype=bool
            ),
            solve_replanned=np.asarray(
                [item["replanned"] for item in solve_diagnostics], dtype=bool
            ),
            solve_phase_index=np.asarray(
                [item["phase_index"] for item in solve_diagnostics], dtype=np.int64
            ),
            solve_elapsed_time=np.asarray(
                [item["elapsed_time"] for item in solve_diagnostics],
                dtype=np.float64,
            ),
            warm_start_shift=np.asarray(
                [item["warm_start_shift"] for item in solve_diagnostics],
                dtype=np.int64,
            ),
            residual_actions_unclipped=residual_unclipped,
            schema_version=np.array(SCHEMA_VERSION),
            task_id=np.array("go2_barrel_roll"),
            roll_direction=np.array(ROLL_DIRECTION_SIGN),
            rollout_seed=np.array(seed),
            sampled_spread=np.array(env._spread),
            success=np.array(True),
            failure_reason=np.array(""),
            go2_sysid_enabled=np.array(True),
            tracking_kp=np.array(TRACKING_KP),
            tracking_kd=np.array(TRACKING_KD),
            final_roll=np.asarray(final_roll),
            final_pitch=np.asarray(final_pitch),
            final_base_height=np.asarray(env.mjData.qpos[2]),
            final_tilt=np.asarray(final_tilt),
            final_roll_progress=np.asarray(env._roll_progress),
            action_clip_fraction=np.asarray(action_clip_fraction),
            mpx_torque_saturation_fraction=np.asarray(mpx_saturation_fraction),
            torque_saturation_fraction=np.asarray(torque_saturation_fraction),
            nonfoot_contact_count=np.asarray(
                np.count_nonzero(nonfoot_contacts != "")
            ),
            final_base_linear_speed=np.asarray(
                np.linalg.norm(env.mjData.qvel[:3])
            ),
            final_base_angular_speed=np.asarray(
                np.linalg.norm(env.mjData.qvel[3:6])
            ),
            final_joint_velocity_norm=np.asarray(
                np.linalg.norm(env.mjData.qvel[6:])
            ),
            terminal_final_hold_streak=np.asarray(env._final_hold_streak),
        )
        data.update({
            f"hold_{name}_valid": np.asarray(values, dtype=bool)
            for name, values in hold_conditions_ctrl.items()
        })
        data.update({
            f"{name}_ctrl": np.asarray(values, dtype=np.float64)
            for name, values in hold_metrics_ctrl.items()
        })
        rollout_xml_path = (
            Path(gym_quadruped.__file__).resolve().parent
            / "robot_model"
            / env.robot_cfg.mjcf_filename
        )
        data.update(
            build_provenance_fields(
                env=env,
                mpx_xml_path=Path(config.model_path),
                rollout_xml_path=rollout_xml_path,
                generator_source_path=Path(__file__),
                generator_command=(
                    generator_command
                    if generator_command is not None
                    else shlex.join([sys.executable, *sys.argv])
                ),
                generator_config=(
                    generator_config
                    if generator_config is not None
                    else {
                        "nominal_spread_zero": nominal_spread_zero,
                        "render": render,
                        "seed": seed,
                    }
                ),
            )
        )
        return data, metrics, trace
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-trajectories", type=int, default=1)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/go2_barrel_roll/v4")
    )
    parser.add_argument("--manifest-filename", default="generation_manifest.jsonl")
    parser.add_argument("--render", action="store_true")
    video_group = parser.add_mutually_exclusive_group()
    video_group.add_argument("--render-video", type=Path)
    video_group.add_argument("--render-video-dir", type=Path)
    video_group.add_argument("--render-accepted-video-dir", type=Path)
    parser.add_argument("--nominal-spread-zero", action="store_true")
    parser.add_argument("--write-commissioning-traces", action="store_true")
    parser.add_argument("--retained-v2-prefix", type=Path)
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    controller = mpc_wrapper.MPCControllerWrapper(config, use_go2_sysid=True)
    accepted = 0
    manifest_path = args.output_dir / args.manifest_filename
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite manifest: {manifest_path}")
    if args.render_video is not None and args.render_video.exists():
        raise FileExistsError(f"refusing to overwrite rendered video: {args.render_video}")
    if args.render_video is not None and args.max_attempts != 1:
        parser.error("--render-video requires --max-attempts=1; use --render-video-dir otherwise")
    if args.render_video_dir is not None:
        args.render_video_dir.mkdir(parents=True, exist_ok=True)
    if args.render_accepted_video_dir is not None:
        args.render_accepted_video_dir.mkdir(parents=True, exist_ok=True)
    generator_command = shlex.join([sys.executable, *sys.argv])
    generator_config = {
        "manifest_filename": args.manifest_filename,
        "max_attempts": args.max_attempts,
        "nominal_spread_zero": args.nominal_spread_zero,
        "num_trajectories": args.num_trajectories,
        "render": args.render,
        "render_video": str(args.render_video) if args.render_video is not None else None,
        "render_video_dir": (
            str(args.render_video_dir)
            if args.render_video_dir is not None
            else None
        ),
        "render_accepted_video_dir": (
            str(args.render_accepted_video_dir)
            if args.render_accepted_video_dir is not None
            else None
        ),
        "start_seed": args.start_seed,
        "write_commissioning_traces": args.write_commissioning_traces,
        "retained_v2_prefix": (
            str(args.retained_v2_prefix)
            if args.retained_v2_prefix is not None
            else None
        ),
    }
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for attempt in range(args.max_attempts):
            seed = args.start_seed + attempt
            attempt_video_path = args.render_video
            accepted_video_path = None
            if args.render_video_dir is not None:
                attempt_video_path = (
                    args.render_video_dir / f"mpc_seed_{seed:07d}.mp4"
                )
                if attempt_video_path.exists():
                    raise FileExistsError(
                        f"refusing to overwrite rendered video: {attempt_video_path}"
                    )
            elif args.render_accepted_video_dir is not None:
                accepted_video_path = (
                    args.render_accepted_video_dir / f"mpc_seed_{seed:07d}.mp4"
                )
                if accepted_video_path.exists():
                    raise FileExistsError(
                        f"refusing to overwrite rendered video: {accepted_video_path}"
                    )
            try:
                data, metrics, trace = generate_attempt(
                    seed,
                    controller,
                    render=args.render,
                    render_video_path=attempt_video_path,
                    render_accepted_video_path=accepted_video_path,
                    nominal_spread_zero=args.nominal_spread_zero,
                    verbose=args.verbose,
                    generator_command=generator_command,
                    generator_config={**generator_config, "seed": seed},
                    retained_v2_prefix=args.retained_v2_prefix,
                )
            except Exception as error:
                data = None
                trace = None
                metrics = {
                    "seed": seed,
                    "accepted": False,
                    "failure_reason": f"{type(error).__name__}: {error}",
                    "go2_sysid_enabled": True,
                }
            metrics, non_finite_metric_fields = _sanitize_metrics(metrics)
            if attempt_video_path is not None and attempt_video_path.exists():
                metrics["video_file"] = str(attempt_video_path)
            if accepted_video_path is not None and accepted_video_path.exists():
                metrics["video_file"] = str(accepted_video_path)
            if non_finite_metric_fields:
                data = None
            if trace is not None and args.write_commissioning_traces:
                trace_path = args.output_dir / (
                    f"commissioning_trace_seed_{seed:06d}.npz"
                )
                temporary_trace = trace_path.with_suffix(".tmp.npz")
                np.savez_compressed(temporary_trace, **trace)
                temporary_trace.replace(trace_path)
            if data is not None:
                path = args.output_dir / (
                    f"go2_barrel_roll_v{SCHEMA_VERSION}_dir_pos_seed_{seed:06d}_"
                    f"ep_{CONTROL_STEPS:03d}.npz"
                )
                atomic_save_npz(path, data)
                metrics["schema_version"] = SCHEMA_VERSION
                metrics["task_id"] = "go2_barrel_roll"
                metrics["trajectory_file"] = path.name
                accepted += 1
            manifest.write(json.dumps(metrics, sort_keys=True, allow_nan=False) + "\n")
            manifest.flush()
            if args.verbose:
                print(json.dumps(metrics, sort_keys=True, allow_nan=False))
            if accepted >= args.num_trajectories:
                break
    if accepted != args.num_trajectories:
        raise SystemExit(
            f"accepted {accepted}/{args.num_trajectories} barrel-roll attempts"
        )


if __name__ == "__main__":
    main()
