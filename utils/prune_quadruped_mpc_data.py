#!/usr/bin/env python3
"""Find quadruped MPC trajectories where the robot torso touches the ground.

This script is intentionally non-destructive: it only prints candidate
trajectory files to stdout. Review the candidates before deleting anything.

Typical usage:

    python utils/prune_quadruped_mpc_data.py \
        --data-dir=data/quadruped_dr/sysid_dyn20_mjlab_10k_no_drift \
        > bad_quadruped_files.txt

View a candidate with the DR trajectory replay script:

    bad_file=data/quadruped_dr/sysid_dyn20_mjlab_10k_no_drift/quadruped_dr_sysid_dyn20_mjlab_seed_203034_ep_1000.npz
    python tests/gen_traj_data_test_quad_dr.py \
        --data-dir="$(dirname "$bad_file")" \
        --filename="$(basename "$bad_file")" \
        --max-steps=500

After reviewing the printed candidates, remove them from the command line:

    xargs -r rm -- < bad_quadruped_files.txt

The detector loads each saved qpos pose into the same flat Go2 MuJoCo model
used by QuadrupedVelocityTrackingEnv and asks MuJoCo for contacts between the
base/torso collision geoms and the world ground geom. It does not replay
torques or alter the trajectory files.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import gym_quadruped
import mujoco
import numpy as np
from gym_quadruped.robot_cfgs import get_robot_config
from gym_quadruped.utils.mujoco.terrain import generate_terrain


@dataclass(frozen=True)
class TorsoGroundHit:
    """First torso-ground contact found in a trajectory."""

    sim_step: int
    time_s: float | None
    base_height_m: float
    geom1: str
    geom2: str
    contact_dist_m: float


def _mj_name(model: mujoco.MjModel, obj_type: mujoco.mjtObj, obj_id: int) -> str:
    name = mujoco.mj_id2name(model, obj_type, obj_id)
    return name if name is not None else f"{obj_type.name}_{obj_id}"


def load_flat_go2_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Build the same flat Go2 model that QuadrupedVelocityTrackingEnv loads."""
    gym_quad_dir = Path(gym_quadruped.__file__).parent
    base_path = gym_quad_dir / "robot_model"
    procedural_assets_path = gym_quad_dir / "utils" / "mujoco" / "assets"

    robot_cfg = get_robot_config(robot_name="go2")
    base_scene_path = procedural_assets_path / "scene_flat.xml"
    scene_env, _terrain_limits = generate_terrain(
        base_scene_path,
        procedural_assets_path,
        robot_cfg.hip_height,
        "flat",
        seed=10,
    )

    robot_xml_path = base_path / robot_cfg.mjcf_filename
    if not robot_xml_path.exists():
        raise FileNotFoundError(f"Go2 robot XML not found: {robot_xml_path}")

    include = ET.Element("include")
    include.attrib["file"] = str(robot_xml_path.absolute().resolve())
    scene_env.getroot().insert(0, include)

    combined_scene_path = Path(tempfile.gettempdir()) / "go2-flat-prune-quadruped.xml"
    scene_env.write(combined_scene_path)

    model = mujoco.MjModel.from_xml_path(str(combined_scene_path))
    data = mujoco.MjData(model)
    return model, data


def collision_geoms_for_body(model: mujoco.MjModel, body_name: str) -> set[int]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body '{body_name}' not found in MuJoCo model")

    return {
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) == body_id
        and int(model.geom_contype[geom_id]) != 0
        and int(model.geom_conaffinity[geom_id]) != 0
    }


def world_collision_geoms(model: mujoco.MjModel) -> set[int]:
    return {
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) == 0
        and int(model.geom_contype[geom_id]) != 0
        and int(model.geom_conaffinity[geom_id]) != 0
    }


def normalized_qpos_array(npz_path: Path, nq: int) -> np.ndarray:
    with np.load(npz_path, allow_pickle=False) as data:
        if "qpos" not in data:
            raise KeyError("missing required array 'qpos'")

        qpos = np.asarray(data["qpos"], dtype=np.float64)

    if qpos.ndim != 2:
        raise ValueError(f"qpos must be 2-D, got shape {qpos.shape}")
    if qpos.shape[0] == nq:
        return qpos
    if qpos.shape[1] == nq:
        return qpos.T

    raise ValueError(f"qpos shape {qpos.shape} does not match model nq={nq}")


def trajectory_time(npz_path: Path, sim_step: int) -> float | None:
    with np.load(npz_path, allow_pickle=False) as data:
        if "time" in data and sim_step < len(data["time"]):
            return float(data["time"][sim_step])
        if "sim_dt" in data:
            return float(data["sim_dt"]) * sim_step
    return None


def find_torso_ground_hit(
    npz_path: Path,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    torso_geom_ids: set[int],
    ground_geom_ids: set[int],
    max_contact_dist: float,
) -> TorsoGroundHit | None:
    qpos = normalized_qpos_array(npz_path, model.nq)

    for sim_step in range(qpos.shape[1]):
        data.qpos[:] = qpos[:, sim_step]
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        mujoco.mj_forward(model, data)

        for contact_idx in range(data.ncon):
            contact = data.contact[contact_idx]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if float(contact.dist) > max_contact_dist:
                continue

            torso_on_ground = (
                geom1 in torso_geom_ids
                and geom2 in ground_geom_ids
            ) or (
                geom2 in torso_geom_ids
                and geom1 in ground_geom_ids
            )
            if not torso_on_ground:
                continue

            return TorsoGroundHit(
                sim_step=sim_step,
                time_s=trajectory_time(npz_path, sim_step),
                base_height_m=float(data.qpos[2]),
                geom1=_mj_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1),
                geom2=_mj_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2),
                contact_dist_m=float(contact.dist),
            )

    return None


def iter_npz_files(data_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.npz" if recursive else "*.npz"
    return sorted(path for path in data_dir.glob(pattern) if path.is_file())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print quadruped trajectory .npz files whose torso/base collision "
            "geoms touch the ground."
        )
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="Directory containing quadruped trajectory .npz files.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for .npz files recursively under --data-dir.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress and first-contact details to stderr.",
    )
    parser.add_argument(
        "--max-contact-dist",
        type=float,
        default=0.0,
        help=(
            "Maximum MuJoCo contact distance to count as a torso-ground hit. "
            "The default, 0.0, requires actual touch/penetration rather than "
            "near-contact inside a geom margin."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.expanduser()

    if not data_dir.exists():
        print(f"error: data directory not found: {data_dir}", file=sys.stderr)
        return 2
    if not data_dir.is_dir():
        print(f"error: --data-dir is not a directory: {data_dir}", file=sys.stderr)
        return 2

    npz_files = iter_npz_files(data_dir, recursive=args.recursive)
    if not npz_files:
        print(f"no .npz files found in {data_dir}", file=sys.stderr)
        return 0

    model, mj_data = load_flat_go2_model()
    torso_geom_ids = collision_geoms_for_body(model, "base")
    ground_geom_ids = world_collision_geoms(model)

    if not torso_geom_ids:
        print("error: no base collision geoms found in model", file=sys.stderr)
        return 2
    if not ground_geom_ids:
        print("error: no world collision geoms found in model", file=sys.stderr)
        return 2

    candidate_count = 0
    error_count = 0

    for index, npz_path in enumerate(npz_files, start=1):
        if args.verbose:
            print(f"[{index}/{len(npz_files)}] checking {npz_path}", file=sys.stderr)

        try:
            hit = find_torso_ground_hit(
                npz_path,
                model,
                mj_data,
                torso_geom_ids,
                ground_geom_ids,
                args.max_contact_dist,
            )
        except Exception as exc:
            error_count += 1
            print(f"warning: skipped {npz_path}: {exc}", file=sys.stderr)
            continue

        if hit is None:
            continue

        candidate_count += 1
        print(npz_path, flush=True)
        if args.verbose:
            time_part = (
                f"{hit.time_s:.4f}s"
                if hit.time_s is not None
                else f"sim_step={hit.sim_step}"
            )
            print(
                "  first torso-ground contact: "
                f"sim_step={hit.sim_step}, time={time_part}, "
                f"base_z={hit.base_height_m:.4f}m, "
                f"contact={hit.geom1}<->{hit.geom2}, "
                f"dist={hit.contact_dist_m:.6f}m",
                file=sys.stderr,
            )

    print(
        f"checked {len(npz_files)} files; found {candidate_count} candidates; "
        f"skipped {error_count} files with errors",
        file=sys.stderr,
    )
    return 1 if error_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
