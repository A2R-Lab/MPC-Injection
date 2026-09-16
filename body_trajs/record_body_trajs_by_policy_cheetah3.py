#!/usr/bin/env python3
"""
Load trained SAC-MPC/TD3-MPC cheetah3 models and record body trajectories
across checkpoints.

The recorded trajectories are saved under mode_traj_data_cheetah3 for later
plotting or analysis. This mirrors record_body_trajs_by_policy_walker.py, but
uses the local three-legged cheetah environment used by run_cheetah_experiments.sh.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Add parent directory to path to import from mpc_rl.
sys.path.insert(0, str(Path(__file__).parent.parent))

from mpc_rl.envs.cheetah3_env import DEFAULT_SPEED_GOAL, make_cheetah3_env
from mpc_rl.sac_mpc.sac_mpc import SAC_MPC
from mpc_rl.td3_mpc.td3_mpc import TD3_MPC


DEFAULT_OUTPUT_DIR = Path(__file__).parent / "mode_traj_data_cheetah3"
DEFAULT_BODY_NAMES = [
    "torso",
    "bthigh",
    "bshin",
    "bfoot",
    "mthigh",
    "mshin",
    "mfoot",
    "fthigh",
    "fshin",
    "ffoot",
]
DEFAULT_FOOT_NAMES = ["bfoot", "mfoot", "ffoot"]
GROUND_GEOM_NAMES = {"ground", "floor"}


def load_config(run_dir: Path) -> dict:
    """Load the configuration from config.json."""
    config_path = run_dir / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def detect_algorithm(run_dir: Path, config: dict) -> str:
    """Infer the algorithm name from config.json or the run directory name."""
    if config.get("algorithm"):
        return str(config["algorithm"])

    run_name = run_dir.name.upper()
    if "SAC-MPC" in run_name:
        return "SAC-MPC"
    if "TD3-MPC" in run_name:
        return "TD3-MPC"
    raise ValueError(f"Could not detect algorithm for run: {run_dir}")


def build_cheetah3_env(speed_goal: float, render_mode=None):
    """Create the cheetah3 environment wrapped exactly as training does."""
    gym_env = make_cheetah3_env(render_mode=render_mode, speed_goal=speed_goal)
    return FlattenObservation(gym_env)


def load_model_and_vecnormalize(
    run_dir: Path,
    config: dict,
    checkpoint_step: int,
    algorithm: str,
):
    """
    Load the trained model and VecNormalize wrapper.

    Args:
        run_dir: Path to the run directory.
        config: Configuration dictionary.
        checkpoint_step: Which checkpoint to load, e.g. 500000.
        algorithm: SAC-MPC or TD3-MPC.

    Returns:
        Tuple of (model, vec_env).
    """
    speed_goal = float(config.get("cheetah3_speed_goal", DEFAULT_SPEED_GOAL))

    vec_env = DummyVecEnv([lambda: build_cheetah3_env(speed_goal=speed_goal)])

    vecnormalize_path = run_dir / "checkpoints" / f"model_vecnormalize_{checkpoint_step}_steps.pkl"
    model_path = run_dir / "checkpoints" / f"model_{checkpoint_step}_steps.zip"

    if not vecnormalize_path.exists():
        raise FileNotFoundError(f"VecNormalize file not found: {vecnormalize_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"  Loading VecNormalize from: {vecnormalize_path.name}")
    vec_env = VecNormalize.load(vecnormalize_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    print(f"  Loading {algorithm} model from: {model_path.name}")
    if algorithm.upper() == "SAC-MPC":
        model = SAC_MPC.load(model_path, env=vec_env)
    elif algorithm.upper() == "TD3-MPC":
        model = TD3_MPC.load(model_path, env=vec_env)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return model, vec_env


def get_geom_name(physics, geom_id: int) -> str | None:
    """Return a MuJoCo geom name, preserving None for unnamed geoms."""
    return physics.model.id2name(geom_id, "geom")


def is_foot_ground_contact(physics, foot_name: str) -> bool:
    """Return whether a foot geom is currently contacting the ground."""
    for i in range(physics.data.ncon):
        contact = physics.data.contact[i]
        geom1_name = get_geom_name(physics, contact.geom1)
        geom2_name = get_geom_name(physics, contact.geom2)
        if geom1_name == foot_name and geom2_name in GROUND_GEOM_NAMES:
            return True
        if geom2_name == foot_name and geom1_name in GROUND_GEOM_NAMES:
            return True
    return False


def run_episode_and_record_trajectories(
    model,
    vec_env,
    body_names: list[str],
    foot_names: list[str],
    max_steps: int = 1000,
    seed: int = 42,
    enable_perturbation: bool = False,
    perturbation_force_x: float = -350.0,
    perturbation_timestep: int = 300,
    perturbation_duration: int = 10,
):
    """
    Run one episode and collect trajectories of each body part.

    Returns a dictionary containing body positions, body orientations, policy
    observations/actions, joint states, foot clearances/contacts, torso height,
    rewards, and perturbation metadata.
    """
    vec_env.seed(seed)
    obs = vec_env.reset()

    base_env = vec_env.envs[0].unwrapped
    physics = base_env._env.physics

    body_positions = {name: [] for name in body_names}
    body_orientations = {name: [] for name in body_names}
    foot_clearances = {name: [] for name in foot_names}
    foot_contacts = {name: [] for name in foot_names}
    rewards = []
    observations = []
    actions = []
    joint_angles = []

    joint_names = [physics.model.id2name(i, "joint") for i in range(physics.model.njnt)]

    done = False
    step_count = 0
    perturbation_applied = False
    perturbation_cleared = False
    perturbation_end_step = perturbation_timestep + perturbation_duration
    torso_body_id = physics.model.name2id("torso", "body")

    while not done and step_count < max_steps:
        if enable_perturbation:
            if perturbation_timestep <= step_count < perturbation_end_step:
                physics.data.xfrc_applied[torso_body_id] = [
                    perturbation_force_x,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
                if not perturbation_applied:
                    print(
                        "    -> Applying perturbation: "
                        f"{perturbation_force_x}N in x-direction for "
                        f"{perturbation_duration} timesteps "
                        f"(starting at step {step_count})"
                    )
                    perturbation_applied = True
            elif perturbation_applied and step_count >= perturbation_end_step and not perturbation_cleared:
                physics.data.xfrc_applied[torso_body_id] = [0, 0, 0, 0, 0, 0]
                print(f"    -> Cleared perturbation at step {step_count}")
                perturbation_cleared = True

        observations.append(obs[0].copy())

        for body_name in body_names:
            body_positions[body_name].append(physics.named.data.xpos[body_name].copy())
            body_orientations[body_name].append(physics.named.data.xquat[body_name].copy())

        for foot_name in foot_names:
            foot_clearances[foot_name].append(physics.named.data.xpos[foot_name][2])
            foot_contacts[foot_name].append(is_foot_ground_contact(physics, foot_name))

        joint_angles.append(physics.data.qpos.copy())

        action, _ = model.predict(obs, deterministic=True)
        actions.append(action[0].copy())

        obs, reward, done, _info = vec_env.step(action)
        rewards.append(reward[0])

        step_count += 1
        done = done[0]

    return {
        "body_positions": {name: np.array(positions) for name, positions in body_positions.items()},
        "body_orientations": {
            name: np.array(orientations) for name, orientations in body_orientations.items()
        },
        "timesteps": step_count,
        "rewards": np.array(rewards),
        "done": done,
        "seed": seed,
        "observations": np.array(observations),
        "actions": np.array(actions),
        "joint_angles": np.array(joint_angles),
        "joint_names": joint_names,
        "foot_clearances": {
            name: np.array(clearances) for name, clearances in foot_clearances.items()
        },
        "foot_contacts": {
            name: np.array(contacts, dtype=bool) for name, contacts in foot_contacts.items()
        },
        "torso_height": np.array(body_positions["torso"])[:, 2],
        "perturbation_applied": perturbation_applied,
        "perturbation_config": {
            "enabled": enable_perturbation,
            "force_x": perturbation_force_x,
            "timestep": perturbation_timestep,
            "duration": perturbation_duration,
        },
    }


def save_trajectory_data(trajectory_data: dict, output_dir: Path, checkpoint_step: int):
    """Save trajectory data to disk as a compressed npz file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"trajectories_step_{checkpoint_step}.npz"

    save_dict = {
        "timesteps": trajectory_data["timesteps"],
        "rewards": trajectory_data["rewards"],
        "done": trajectory_data["done"],
        "seed": trajectory_data["seed"],
        "observations": trajectory_data["observations"],
        "actions": trajectory_data["actions"],
        "joint_angles": trajectory_data["joint_angles"],
        "joint_names": trajectory_data["joint_names"],
    }

    for body_name, positions in trajectory_data["body_positions"].items():
        save_dict[f"pos_{body_name}"] = positions

    for body_name, orientations in trajectory_data["body_orientations"].items():
        save_dict[f"quat_{body_name}"] = orientations

    for foot_name, clearances in trajectory_data["foot_clearances"].items():
        save_dict[f"clearance_{foot_name}"] = clearances

    for foot_name, contacts in trajectory_data["foot_contacts"].items():
        save_dict[f"contact_{foot_name}"] = contacts
        save_dict[f"contact_pct_{foot_name}"] = (contacts.sum() / len(contacts)) * 100

    save_dict["torso_height"] = trajectory_data["torso_height"]
    save_dict["torso_height_mean"] = trajectory_data["torso_height"].mean()
    save_dict["torso_height_std"] = trajectory_data["torso_height"].std()

    save_dict["perturbation_applied"] = trajectory_data["perturbation_applied"]
    save_dict["perturbation_enabled"] = trajectory_data["perturbation_config"]["enabled"]
    save_dict["perturbation_force_x"] = trajectory_data["perturbation_config"]["force_x"]
    save_dict["perturbation_timestep"] = trajectory_data["perturbation_config"]["timestep"]
    save_dict["perturbation_duration"] = trajectory_data["perturbation_config"]["duration"]

    np.savez_compressed(output_file, **save_dict)
    print(f"  Saved trajectory data to: {output_file}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Record body trajectories for a trained cheetah3 SAC-MPC/TD3-MPC run."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-checkpoint", type=int, default=25_000)
    parser.add_argument("--end-checkpoint", type=int, default=500_000)
    parser.add_argument("--checkpoint-step", type=int, default=25_000)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--algorithm", choices=["SAC-MPC", "TD3-MPC"], default=None)
    parser.add_argument("--enable-perturbation", action="store_true")
    parser.add_argument("--perturbation-force-x", type=float, default=-350.0)
    parser.add_argument("--perturbation-timestep", type=int, default=300)
    parser.add_argument("--perturbation-duration", type=int, default=10)
    return parser.parse_args()


def main():
    """Iterate through checkpoints and record cheetah3 body trajectories."""
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()

    print("=" * 80)
    print("Recording Cheetah3 Body Trajectories Across Checkpoints")
    print("=" * 80)
    print(f"\nRun directory: {run_dir}")

    config = load_config(run_dir)
    algorithm = args.algorithm or detect_algorithm(run_dir, config)
    checkpoints = list(
        range(args.start_checkpoint, args.end_checkpoint + 1, args.checkpoint_step)
    )

    print(f"Environment: {config.get('domain')}-{config.get('task')}")
    print(f"Algorithm: {algorithm}")
    print(f"Cheetah3 speed goal: {config.get('cheetah3_speed_goal', DEFAULT_SPEED_GOAL)}")
    print(
        f"\nProcessing {len(checkpoints)} checkpoints: "
        f"{args.start_checkpoint} to {args.end_checkpoint} "
        f"(step: {args.checkpoint_step})"
    )
    print(f"Max steps per episode: {args.max_steps}")
    print(f"Random seed: {args.seed}")
    print(f"Body parts tracked: {', '.join(DEFAULT_BODY_NAMES)}")
    print("\nPerturbation settings:")
    print(f"  Enabled: {args.enable_perturbation}")
    if args.enable_perturbation:
        print(f"  Force (x-axis): {args.perturbation_force_x} N")
        print(f"  Apply at timestep: {args.perturbation_timestep}")
        print(f"  Duration: {args.perturbation_duration} timesteps")

    output_subdir = args.output_dir / run_dir.name
    print(f"\nOutput directory: {output_subdir}")

    print("\n" + "=" * 80)
    for i, checkpoint_step in enumerate(checkpoints, 1):
        print(f"\n[{i}/{len(checkpoints)}] Processing checkpoint: {checkpoint_step}")
        print("-" * 80)

        try:
            model, vec_env = load_model_and_vecnormalize(
                run_dir,
                config,
                checkpoint_step,
                algorithm=algorithm,
            )

            print(f"  Running episode (seed={args.seed})...")
            trajectory_data = run_episode_and_record_trajectories(
                model,
                vec_env,
                DEFAULT_BODY_NAMES,
                DEFAULT_FOOT_NAMES,
                max_steps=args.max_steps,
                seed=args.seed,
                enable_perturbation=args.enable_perturbation,
                perturbation_force_x=args.perturbation_force_x,
                perturbation_timestep=args.perturbation_timestep,
                perturbation_duration=args.perturbation_duration,
            )

            print(f"  Episode completed: {trajectory_data['timesteps']} steps")
            print(f"  Total reward: {trajectory_data['rewards'].sum():.2f}")
            print(f"  Early termination: {trajectory_data['done']}")

            for foot_name in DEFAULT_FOOT_NAMES:
                contact_pct = (
                    trajectory_data["foot_contacts"][foot_name].sum()
                    / trajectory_data["timesteps"]
                ) * 100
                print(f"  {foot_name} contact: {contact_pct:.1f}%")

            torso_mean_height = trajectory_data["torso_height"].mean()
            torso_std_height = trajectory_data["torso_height"].std()
            print(f"  Torso height: {torso_mean_height:.3f} +/- {torso_std_height:.3f} m")

            save_trajectory_data(trajectory_data, output_subdir, checkpoint_step)
            vec_env.close()

        except FileNotFoundError as e:
            print(f"  WARNING: Skipping checkpoint {checkpoint_step} - {e}")
            continue
        except Exception as e:
            print(f"  ERROR: Failed to process checkpoint {checkpoint_step}")
            print(f"  Error details: {e}")
            import traceback

            traceback.print_exc()
            continue

    print("\n" + "=" * 80)
    print("Processing complete!")
    print(f"All trajectory data saved to: {output_subdir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
