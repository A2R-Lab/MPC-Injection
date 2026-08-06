"""Commission direct Go2 barrel-roll transitions with 50 Hz MPC replanning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import mpx.config.config_barrel_roll as config
import mpx.utils.mpc_wrapper as mpc_wrapper
from mpc_rl.envs.barrel_roll_common import (
    CONTROL_DT,
    CONTROL_STEPS,
    MANEUVER_HORIZON,
    ROLL_DIRECTION_SIGN,
    desired_roll_at_time,
)
from mpc_rl.envs.barrel_roll_env import QuadrupedBarrelRollEnv


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


def generate_attempt(
    seed: int,
    mpc: mpc_wrapper.MPCControllerWrapper,
    *,
    render: bool = False,
    nominal_spread_zero: bool = False,
    verbose: int = 1,
):
    """Execute one seeded receding-horizon attempt in the RL task plant."""
    env = QuadrupedBarrelRollEnv(
        render_mode="human" if render else None,
        use_go2_sysid=True,
    )
    try:
        reset_options = {"spread": 0.0} if nominal_spread_zero else None
        obs, _ = env.reset(seed=seed, options=reset_options)
        initial_qpos = env.mjData.qpos.copy()
        initial_qvel = env.mjData.qvel.copy()
        mpc.reset(initial_qpos, initial_qvel)
        if render:
            env.render()

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
        classifier_results = []
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

        for control_step in range(CONTROL_STEPS):
            elapsed_time = control_step * CONTROL_DT
            stop_replanning_after_update = bool(
                replanning_active
                and elapsed_time >= config.BARREL_REPLAN_EARLIEST_STOP_TIME
                and np.all(env._get_foot_contacts())
                and env._stability_count >= config.BARREL_REPLAN_STABLE_STEPS
            )
            transitions["policy_obs"].append(obs["policy"].copy())
            transitions["privileged_obs"].append(obs["privileged"].copy())

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
                if (
                    env._physics_failure_reason is not None
                    and env._physics_failure_reason.startswith(
                        "non_foot_ground_contact:"
                    )
                ):
                    contact_name = env._physics_failure_reason.split(":", 1)[1]
                    nonfoot_contacts[sim_step] = contact_name
                    if first_nonfoot_contact is None:
                        first_nonfoot_contact = contact_name
                        first_nonfoot_time = float(env.mjData.time)
                min_base_height = min(min_base_height, float(env.mjData.qpos[2]))

                if render:
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
            classifier_results.append(env._terminal_success())

            transitions["actions"].append(action.copy())
            transitions["rewards"].append(reward)
            transitions["terminated_ctrl"].append(terminated)
            transitions["truncated_ctrl"].append(False)
            transitions["next_policy_obs"].append(next_obs["policy"].copy())
            transitions["next_privileged_obs"].append(next_obs["privileged"].copy())
            obs = next_obs

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
        final_contacts = env._filtered_barrel_contacts.copy()
        metrics = {
            "seed": seed,
            "accepted": success,
            "failure_reason": failure_reason,
            "completed_control_steps": completed_steps,
            "roll_progress": float(env._roll_progress),
            "final_roll": final_roll,
            "final_pitch": final_pitch,
            "final_base_height": float(env.mjData.qpos[2]),
            "stable_contact_streak": int(env._stability_count),
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
        }
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
        }
        if not success:
            return None, metrics, trace

        data = {key: np.asarray(value) for key, value in transitions.items()}
        data.update(
            qpos=qpos,
            qvel=qvel,
            tau_applied=tau,
            tau_mpx=tau_mpx,
            q_des=q_des,
            X_updates=np.asarray(X_updates),
            U_updates=np.asarray(U_updates),
            solve_seconds=solve_times,
            solve_objective_norm_sq=np.asarray(
                [item["final_objective_norm_sq"] for item in solve_diagnostics]
            ),
            solve_constraint_norm_sq=np.asarray(
                [item["final_constraint_norm_sq"] for item in solve_diagnostics]
            ),
            residual_actions_unclipped=residual_unclipped,
            schema_version=np.array(1),
            task_id=np.array("go2_barrel_roll"),
            roll_direction=np.array(ROLL_DIRECTION_SIGN),
            rollout_seed=np.array(seed),
            sampled_spread=np.array(env._spread),
            success=np.array(True),
            go2_sysid_enabled=np.array(True),
            tracking_kp=np.array(TRACKING_KP),
            tracking_kd=np.array(TRACKING_KD),
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
        "--output-dir", type=Path, default=Path("data/go2_barrel_roll/v1")
    )
    parser.add_argument("--manifest-filename", default="generation_manifest.jsonl")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--nominal-spread-zero", action="store_true")
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    controller = mpc_wrapper.MPCControllerWrapper(config, use_go2_sysid=True)
    accepted = 0
    manifest_path = args.output_dir / args.manifest_filename
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for attempt in range(args.max_attempts):
            seed = args.start_seed + attempt
            try:
                data, metrics, trace = generate_attempt(
                    seed,
                    controller,
                    render=args.render,
                    nominal_spread_zero=args.nominal_spread_zero,
                    verbose=args.verbose,
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
            if non_finite_metric_fields:
                data = None
            manifest.write(json.dumps(metrics, sort_keys=True, allow_nan=False) + "\n")
            manifest.flush()
            if args.verbose:
                print(json.dumps(metrics, sort_keys=True, allow_nan=False))
            if trace is not None:
                trace_path = args.output_dir / (
                    f"commissioning_trace_seed_{seed:06d}.npz"
                )
                temporary_trace = trace_path.with_suffix(".tmp.npz")
                np.savez_compressed(temporary_trace, **trace)
                temporary_trace.replace(trace_path)
            if data is not None:
                path = args.output_dir / (
                    f"go2_barrel_roll_v1_dir_pos_seed_{seed:06d}_"
                    f"ep_{CONTROL_STEPS:03d}.npz"
                )
                temporary = path.with_suffix(".tmp.npz")
                np.savez_compressed(temporary, **data)
                temporary.replace(path)
                accepted += 1
            if accepted >= args.num_trajectories:
                break
    if accepted != args.num_trajectories:
        raise SystemExit(
            f"accepted {accepted}/{args.num_trajectories} barrel-roll attempts"
        )


if __name__ == "__main__":
    main()
