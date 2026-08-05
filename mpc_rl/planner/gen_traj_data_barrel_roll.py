"""Commissioning generator for direct Go2 barrel-roll MPC transitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from timeit import default_timer as timer

import mujoco
import numpy as np

import mpx.config.config_barrel_roll as config
import mpx.utils.mpc_wrapper as mpc_wrapper
from mpc_rl.envs.barrel_roll_common import CONTROL_STEPS, ROLL_DIRECTION_SIGN
from mpc_rl.envs.barrel_roll_env import QuadrupedBarrelRollEnv


def generate_attempt(seed: int, mpc: mpc_wrapper.MPCControllerWrapper, *, render: bool = False):
    env = QuadrupedBarrelRollEnv(render_mode="human" if render else None)
    try:
        obs, _ = env.reset(seed=seed)
        initial_qpos, initial_qvel = env.mjData.qpos.copy(), env.mjData.qvel.copy()
        mpc.reset(initial_qpos, initial_qvel)
        started = timer()
        X, U, reference, _ = mpc.runOffline(initial_qpos, initial_qvel)
        X, U = np.asarray(X), np.asarray(U)
        if X.shape[0] != 101 or U.shape != (100, 12) or not np.isfinite(X).all() or not np.isfinite(U).all():
            return None, {"seed": seed, "accepted": False, "failure_reason": "non_finite_solver_output"}
        transitions = {key: [] for key in ("policy_obs", "next_policy_obs", "privileged_obs", "next_privileged_obs", "actions", "rewards", "terminated_ctrl", "truncated_ctrl")}
        qpos = np.empty((19, 201)); qvel = np.empty((18, 201)); tau = np.empty((12, 200)); tau_mpx = np.empty((12, 200)); q_des = np.empty((12, 200))
        qpos[:, 0], qvel[:, 0] = initial_qpos, initial_qvel
        for control in range(CONTROL_STEPS):
            transitions["policy_obs"].append(obs["policy"]); transitions["privileged_obs"].append(obs["privileged"])
            first_tau = None
            for substep in range(4):
                node = control * 2 + substep // 2
                desired, feedforward = X[node, 7:19], U[node]
                current_q, current_dq = env.mjData.qpos[7:].copy(), env.mjData.qvel[6:].copy()
                applied = np.clip(feedforward + 10.0 * (desired - current_q) - 2.0 * current_dq, env.torque_limits[:, 0], env.torque_limits[:, 1])
                if first_tau is None: first_tau = applied.copy()
                sim = control * 4 + substep
                tau[:, sim], tau_mpx[:, sim], q_des[:, sim] = applied, feedforward, desired
                env.mjData.ctrl[:] = applied; mujoco.mj_step(env.mjModel, env.mjData); env._after_physics_substep()
                qpos[:, sim + 1], qvel[:, sim + 1] = env.mjData.qpos, env.mjData.qvel
            env._step_count += 1; env._update_feet_air_time()
            action_unclipped = ((env.mjData.qpos[7:] + (first_tau + env.kd * env.mjData.qvel[6:]) / env.kp - env.default_joint_pos) / env.action_scale)
            action = np.clip(action_unclipped, -1.0, 1.0)
            env._last_action = action.copy(); terminated = env._check_termination(); reward = env._compute_reward(action, terminated); next_obs = env._get_obs()
            transitions["actions"].append(action); transitions["rewards"].append(reward); transitions["terminated_ctrl"].append(terminated); transitions["truncated_ctrl"].append(False); transitions["next_policy_obs"].append(next_obs["policy"]); transitions["next_privileged_obs"].append(next_obs["privileged"])
            obs = next_obs
            if terminated: break
        metrics = {"seed": seed, "accepted": bool(env._failure_reason is None and len(transitions["rewards"]) == CONTROL_STEPS), "failure_reason": env._failure_reason, "roll_progress": env._roll_progress, "solve_seconds": timer() - started, "diagnostics": mpc.last_offline_diagnostics}
        if not metrics["accepted"]: return None, metrics
        data = {key: np.asarray(value) for key, value in transitions.items()}
        data.update(qpos=qpos, qvel=qvel, tau_applied=tau, tau_mpx=tau_mpx, q_des=q_des, X=X, U=U, schema_version=np.array(1), task_id=np.array("go2_barrel_roll"), roll_direction=np.array(ROLL_DIRECTION_SIGN), rollout_seed=np.array(seed), sampled_spread=np.array(env._spread), success=np.array(True))
        return data, metrics
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--num-trajectories", type=int, default=1); parser.add_argument("--start-seed", type=int, default=0); parser.add_argument("--max-attempts", type=int, default=1); parser.add_argument("--output-dir", type=Path, default=Path("data/go2_barrel_roll/v1")); parser.add_argument("--manifest-filename", default="generation_manifest.jsonl"); parser.add_argument("--render", action="store_true"); parser.add_argument("--nominal-spread-zero", action="store_true"); args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True); mpc = mpc_wrapper.MPCControllerWrapper(config, use_go2_sysid=False); accepted = 0
    with (args.output_dir / args.manifest_filename).open("w") as manifest:
        for attempt in range(args.max_attempts):
            data, metrics = generate_attempt(args.start_seed + attempt, mpc, render=args.render)
            manifest.write(json.dumps(metrics, default=str) + "\n"); manifest.flush()
            if data is not None:
                path = args.output_dir / f"go2_barrel_roll_v1_dir_pos_seed_{args.start_seed + attempt:06d}_ep_050.npz"; temporary = path.with_suffix(".tmp.npz"); np.savez_compressed(temporary, **data); temporary.replace(path); accepted += 1
            if accepted >= args.num_trajectories: break
    if accepted != args.num_trajectories: raise SystemExit(f"accepted {accepted}/{args.num_trajectories} barrel-roll attempts")


if __name__ == "__main__": main()
