"""Evaluate and record the working nominal 0.5 m/s MPX bounding controller."""

import argparse
import json
from pathlib import Path

import numpy as np

import mpx.utils.mpc_wrapper as mpc_wrapper
from mpc_rl.envs.action_interfaces import MPX_BOUND_ACTION_INTERFACE_ID
from mpc_rl.planner.gen_traj_data_mpx_dr import (
    _attempt_evidence,
    _controller_config_with_overrides,
    generate_trajectory,
)
from mpc_rl.planner.mpx_bounding_data import measured_bound_metrics


WORKING_CONTROLLER = {
    "target_command": np.array([0.5, 0.0, 0.0], dtype=np.float64),
    "command_ramp_control_steps": 50,
    "gait": "bound_front_first",
    "duty_factor": 0.65,
    "step_frequency_hz": 2.5,
    "step_height_m": 0.05,
    "mpx_qrot_pitch_cost": 25_000.0,
    "mpx_qomega_pitch_cost": 1_000.0,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record nominal 0.5 m/s MPX bounding evaluations"
    )
    parser.add_argument("--start-seed", type=int, default=1000)
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--episode-length", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=1)
    return parser


def _make_controller():
    controller_config = _controller_config_with_overrides(
        WORKING_CONTROLLER["mpx_qrot_pitch_cost"],
        WORKING_CONTROLLER["mpx_qomega_pitch_cost"],
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    controller = mpc_wrapper.MPCControllerWrapper(
        controller_config,
        gait=WORKING_CONTROLLER["gait"],
        duty_factor=WORKING_CONTROLLER["duty_factor"],
        step_frequency_hz=WORKING_CONTROLLER["step_frequency_hz"],
        step_height_m=WORKING_CONTROLLER["step_height_m"],
        enable_planned_contact_diagnostics=True,
    )
    controller.robot_height = controller_config.robot_height
    return controller


def _summarize(seed: int, trajectory: dict, video_path: Path) -> dict:
    ramp_steps = WORKING_CONTROLLER["command_ramp_control_steps"]
    completed_steps = int(trajectory["completed_control_steps"])
    evidence = _attempt_evidence(trajectory, ramp_control_steps=ramp_steps)
    hold_velocity = np.asarray(trajectory["next_qvel_ctrl"], dtype=np.float64)[
        ramp_steps:completed_steps, 0
    ]
    contacts = np.asarray(trajectory["foot_contacts_substeps"], dtype=bool).reshape(
        -1, 4
    )
    hold_contacts = contacts[ramp_steps * 4 : completed_steps * 4]
    bound = (
        measured_bound_metrics(hold_contacts, contact_debounce_substeps=2)
        if hold_contacts.size
        else None
    )
    return {
        "seed": seed,
        "video": video_path.name,
        "completed_control_steps": completed_steps,
        "fell": bool(trajectory["fell"]),
        "failure_reason": str(trajectory["failure_reason"]),
        "mean_hold_forward_velocity_m_per_s": (
            float(np.mean(hold_velocity)) if hold_velocity.size else None
        ),
        "action_clipping_fraction": evidence["action_clipping"][
            "element_fraction"
        ],
        "max_normalized_clip_excess": evidence["action_clipping"][
            "max_magnitude"
        ],
        "minimum_base_height_m": evidence["posture"]["min_base_height_m"],
        "front_pair_agreement": (
            bound["front_pair_agreement"] if bound is not None else None
        ),
        "rear_pair_agreement": (
            bound["rear_pair_agreement"] if bound is not None else None
        ),
        "paired_interval_alternation_fraction": (
            bound["paired_interval_alternation_fraction"]
            if bound is not None
            else None
        ),
        "complete_front_rear_cycles": (
            bound["complete_front_rear_cycles"] if bound is not None else 0
        ),
        "diagonal_lateral_only_fraction": (
            bound["diagonal_lateral_only_fraction"]
            if bound is not None
            else None
        ),
    }


def main() -> None:
    args = _parser().parse_args()
    if args.num_seeds <= 0:
        raise ValueError("--num-seeds must be positive")
    if args.episode_length <= 0:
        raise ValueError("--episode-length must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    controller = _make_controller()
    results = []

    for seed in range(args.start_seed, args.start_seed + args.num_seeds):
        video_path = args.output_dir / f"mpx_bound_vx0p5_seed_{seed}.mp4"
        trajectory = generate_trajectory(
            seed,
            domain_rand_config_type="disabled",
            mpc=controller,
            episode_length=args.episode_length,
            verbose=args.verbose,
            video_path=video_path,
            fixed_command=WORKING_CONTROLLER["target_command"],
            command_ramp_control_steps=WORKING_CONTROLLER[
                "command_ramp_control_steps"
            ],
            gait=WORKING_CONTROLLER["gait"],
            duty_factor=WORKING_CONTROLLER["duty_factor"],
            step_frequency_hz=WORKING_CONTROLLER["step_frequency_hz"],
            step_height_m=WORKING_CONTROLLER["step_height_m"],
            mpx_qrot_pitch_cost=WORKING_CONTROLLER["mpx_qrot_pitch_cost"],
            mpx_qomega_pitch_cost=WORKING_CONTROLLER["mpx_qomega_pitch_cost"],
            action_interface_id=MPX_BOUND_ACTION_INTERFACE_ID,
        )
        result = _summarize(seed, trajectory, video_path)
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    summary = {
        "controller": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in WORKING_CONTROLLER.items()
        },
        "episode_length": args.episode_length,
        "start_seed": args.start_seed,
        "num_seeds": args.num_seeds,
        "successful_full_horizon_rollouts": sum(
            not result["fell"]
            and result["completed_control_steps"] == args.episode_length
            for result in results
        ),
        "results": results,
    }
    summary_path = args.output_dir / "evaluation_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
