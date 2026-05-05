"""Shared helpers for the three-legged cheetah task."""

from __future__ import annotations

import mujoco
import numpy as np


MIN_GEOM_GROUND_CLEARANCE = 0.015
MIN_TORSO_UPRIGHT_Z = 0.8
MIN_TORSO_HEIGHT = 0.45
MAX_INIT_SAMPLE_ATTEMPTS = 10000


def _geom_min_z(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> float:
    """Return an approximate world-frame lower bound for a geom."""
    geom_type = model.geom_type[geom_id]
    size = model.geom_size[geom_id]
    center_z = float(data.geom_xpos[geom_id, 2])

    if geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
        return float("-inf")
    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        return center_z - float(size[0])
    if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        axis = data.geom_xmat[geom_id].reshape(3, 3)[:, 2]
        return center_z - float(size[0]) - abs(float(axis[2])) * float(size[1])

    # Fallback for geoms not currently used by cheetah3.
    return center_z - float(model.geom_rbound[geom_id])


def cheetah3_pose_diagnostics(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    qvel: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute validity diagnostics for a cheetah3 state."""
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    if qvel is not None:
        data.qvel[:] = qvel
    else:
        data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    torso_xmat = data.xmat[torso_id].reshape(3, 3)
    torso_upright_z = float(torso_xmat[:, 2][2])
    torso_height = float(data.xpos[torso_id, 2])

    min_geom_z = float("inf")
    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if geom_name == "ground":
            continue
        min_geom_z = min(min_geom_z, _geom_min_z(model, data, geom_id))

    return {
        "min_geom_z": min_geom_z,
        "torso_upright_z": torso_upright_z,
        "torso_height": torso_height,
    }


def is_valid_cheetah3_initial_state(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    qvel: np.ndarray | None = None,
    min_geom_z: float = MIN_GEOM_GROUND_CLEARANCE,
    min_torso_upright_z: float = MIN_TORSO_UPRIGHT_Z,
    min_torso_height: float = MIN_TORSO_HEIGHT,
) -> bool:
    """Return True if a cheetah3 initial state is upright and above ground."""
    diagnostics = cheetah3_pose_diagnostics(model, qpos, qvel)
    return (
        diagnostics["min_geom_z"] >= min_geom_z
        and diagnostics["torso_upright_z"] >= min_torso_upright_z
        and diagnostics["torso_height"] >= min_torso_height
    )


def sample_valid_cheetah3_initial_state(
    model: mujoco.MjModel,
    rng,
    max_attempts: int = MAX_INIT_SAMPLE_ATTEMPTS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample a walker-like cheetah3 initial state that is not clipped into ground.

    Root slide joints stay at XML defaults. The unlimited root hinge and all
    limited leg hinges are randomized, but samples that put the body below the
    floor or in a strongly non-upright orientation are rejected.
    """
    if model.nq != model.njnt:
        raise ValueError("cheetah3 initialization assumes one qpos per joint.")

    for _ in range(max_attempts):
        qpos = model.qpos0.copy()
        qvel = np.zeros(model.nv)

        for joint_id in range(model.njnt):
            qpos_adr = model.jnt_qposadr[joint_id]
            joint_type = model.jnt_type[joint_id]
            is_limited = bool(model.jnt_limited[joint_id])

            if is_limited:
                low, high = model.jnt_range[joint_id]
                qpos[qpos_adr] = rng.uniform(low, high)
            elif joint_type == mujoco.mjtJoint.mjJNT_HINGE:
                qpos[qpos_adr] = rng.uniform(-np.pi, np.pi)

        if is_valid_cheetah3_initial_state(model, qpos, qvel):
            return qpos, qvel

    raise RuntimeError(
        f"Failed to sample a valid cheetah3 initial state after {max_attempts} attempts."
    )
