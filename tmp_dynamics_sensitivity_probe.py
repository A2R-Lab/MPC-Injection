from pathlib import Path

import mujoco
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from mpc_rl.envs.domain_randomization import DomainRandomizationConfig
from mpc_rl.envs.velocity_tracking_env import QuadrupedVelocityTrackingEnv


POLICIES = {
    "old_success_256_s100": Path(
        "logs/quadruped_domain_rand_mpc_dr_new_data_test/"
        "SAC-MPC-default_no_push/"
        "quadruped-velocity_tracking-SAC-MPC-20260417-134426-"
        "percentage-25pct-seed100-env256-drdefault_no_push"
    ),
    "sysid_512_s150": Path(
        "logs/quadruped_domain_rand_mpc_sys_id/"
        "SAC-MPC-sysid_floor_sensing_no_push/"
        "quadruped-velocity_tracking-SAC-MPC-20260424-001153-"
        "percentage-25pct-seed150-env512-drsysid_floor_sensing_no_push"
    ),
}

MODES = ("sysid", "raw_xml", "midway")
EVAL_SEEDS = [11, 22, 33]
VX = 0.5
HORIZON = 400


def build_raw_joint_dynamics(env: QuadrupedVelocityTrackingEnv) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import gym_quadruped

    gym_quad_dir = Path(gym_quadruped.__file__).parent
    robot_xml_path = gym_quad_dir / "robot_model" / env.robot_cfg.mjcf_filename
    raw_model = mujoco.MjModel.from_xml_path(str(robot_xml_path))
    return (
        raw_model.dof_armature.copy(),
        raw_model.dof_frictionloss.copy(),
        raw_model.dof_damping.copy(),
    )


def apply_joint_dynamics_mode(env: QuadrupedVelocityTrackingEnv, mode: str) -> None:
    if mode == "sysid":
        return

    raw_armature, raw_frictionloss, raw_damping = build_raw_joint_dynamics(env)
    sysid_armature = env.mjModel.dof_armature.copy()
    sysid_frictionloss = env.mjModel.dof_frictionloss.copy()
    sysid_damping = env.mjModel.dof_damping.copy()

    if mode == "raw_xml":
        env.mjModel.dof_armature[:] = raw_armature
        env.mjModel.dof_frictionloss[:] = raw_frictionloss
        env.mjModel.dof_damping[:] = raw_damping
    elif mode == "midway":
        env.mjModel.dof_armature[:] = 0.5 * (raw_armature + sysid_armature)
        env.mjModel.dof_frictionloss[:] = 0.5 * (raw_frictionloss + sysid_frictionloss)
        env.mjModel.dof_damping[:] = 0.5 * (raw_damping + sysid_damping)
    else:
        raise ValueError(mode)

    env._nominal_dof_armature = env.mjModel.dof_armature.copy()
    env._nominal_dof_frictionloss = env.mjModel.dof_frictionloss.copy()
    env._nominal_dof_damping = env.mjModel.dof_damping.copy()


def make_env(mode: str) -> QuadrupedVelocityTrackingEnv:
    env = QuadrupedVelocityTrackingEnv(
        robot="go2",
        scene="flat",
        domain_rand_cfg=DomainRandomizationConfig.disabled(),
        apply_startup_domain_rand_on_init=False,
        simple_reward=True,
    )
    apply_joint_dynamics_mode(env, mode)
    return env


def evaluate_policy(policy_dir: Path, mode: str, seed: int) -> dict[str, float]:
    base_env = make_env(mode)
    vec = DummyVecEnv([lambda: base_env])
    vec = VecNormalize.load(str(policy_dir / "vec_normalize.pkl"), vec)
    vec.training = False
    vec.norm_reward = False
    model = SAC.load(str(policy_dir / "final_model.zip"), env=vec)

    base_env.reset(seed=seed)
    obs = vec.reset()
    base_env.set_commands(vx=VX, vy=0.0, wz=0.0)
    obs_dict = base_env._get_obs()
    obs = {key: np.array([obs_dict[key]]) for key in obs_dict}
    obs = vec.normalize_obs(obs)

    rewards: list[float] = []
    vxs: list[float] = []
    done = False
    steps = 0
    while steps < HORIZON and not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done_arr, _info = vec.step(action)
        done = bool(done_arr[0])
        rewards.append(float(reward[0]))
        vxs.append(float(base_env._base_lin_vel_body()[0]))
        steps += 1

    vec.close()
    return {
        "steps": float(steps),
        "survived": float(steps >= HORIZON),
        "mean_reward": float(np.mean(rewards)) if rewards else float("nan"),
        "mean_vx": float(np.mean(vxs)) if vxs else float("nan"),
    }


def main() -> None:
    for policy_name, policy_dir in POLICIES.items():
        print(f"\nPOLICY {policy_name}")
        for mode in MODES:
            results = [evaluate_policy(policy_dir, mode, seed) for seed in EVAL_SEEDS]
            print(
                mode,
                "survival",
                round(float(np.mean([r["survived"] for r in results])), 3),
                "steps",
                round(float(np.mean([r["steps"] for r in results])), 1),
                "mean_reward",
                round(float(np.mean([r["mean_reward"] for r in results])), 3),
                "mean_vx",
                round(float(np.mean([r["mean_vx"] for r in results])), 3),
            )


if __name__ == "__main__":
    main()
