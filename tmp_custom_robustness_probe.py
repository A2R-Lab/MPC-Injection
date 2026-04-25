from pathlib import Path

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

SCENARIOS = {
    "nominal_clean": DomainRandomizationConfig.disabled(),
    "floor_only": DomainRandomizationConfig.sysid_floor_only_no_push(),
    "sensing_only": DomainRandomizationConfig(
        enable=True,
        friction_range=(0.0, 0.0),
        friction_target_geom_names=None,
        added_mass_range=(0.0, 0.0),
        com_displacement_range=(0.0, 0.0),
        encoder_bias_range=(-0.015, 0.015),
        kp_scale_range=(1.0, 1.0),
        kd_scale_range=(1.0, 1.0),
        joint_damping_scale_range=(1.0, 1.0),
        joint_armature_scale_range=(1.0, 1.0),
        joint_friction_range=(0.0, 0.0),
        motor_strength_range=(1.0, 1.0),
        obs_noise_level=1.0,
        obs_noise_scales={
            "joint_pos": 0.01,
            "joint_vel": 1.5,
            "ang_vel": 0.2,
            "gravity": 0.05,
        },
        push_robots=False,
    ),
    "com_only": DomainRandomizationConfig(
        enable=True,
        friction_range=(0.0, 0.0),
        friction_target_geom_names=None,
        added_mass_range=(0.0, 0.0),
        com_displacement_range=(-0.05, 0.05),
        encoder_bias_range=(0.0, 0.0),
        kp_scale_range=(1.0, 1.0),
        kd_scale_range=(1.0, 1.0),
        joint_damping_scale_range=(1.0, 1.0),
        joint_armature_scale_range=(1.0, 1.0),
        joint_friction_range=(0.0, 0.0),
        motor_strength_range=(1.0, 1.0),
        obs_noise_level=0.0,
        obs_noise_scales={
            "joint_pos": 0.01,
            "joint_vel": 1.5,
            "ang_vel": 0.2,
            "gravity": 0.05,
        },
        push_robots=False,
    ),
    "all_geom_friction_only": DomainRandomizationConfig(
        enable=True,
        friction_range=(0.3, 1.2),
        friction_target_geom_names=None,
        added_mass_range=(0.0, 0.0),
        com_displacement_range=(0.0, 0.0),
        encoder_bias_range=(0.0, 0.0),
        kp_scale_range=(1.0, 1.0),
        kd_scale_range=(1.0, 1.0),
        joint_damping_scale_range=(1.0, 1.0),
        joint_armature_scale_range=(1.0, 1.0),
        joint_friction_range=(0.0, 0.0),
        motor_strength_range=(1.0, 1.0),
        obs_noise_level=0.0,
        obs_noise_scales={
            "joint_pos": 0.01,
            "joint_vel": 1.5,
            "ang_vel": 0.2,
            "gravity": 0.05,
        },
        push_robots=False,
    ),
    "default_no_push": DomainRandomizationConfig.default_no_push(),
}

EVAL_SEEDS = [11, 22, 33]
VX = 0.5
HORIZON = 400


def make_env_from_cfg(dr_cfg: DomainRandomizationConfig, dr_seed: int) -> QuadrupedVelocityTrackingEnv:
    env = QuadrupedVelocityTrackingEnv(
        robot="go2",
        scene="flat",
        domain_rand_cfg=dr_cfg,
        apply_startup_domain_rand_on_init=False,
        simple_reward=True,
    )
    if dr_cfg.enable:
        patch = env.sample_startup_domain_rand_bundle(
            rng=np.random.RandomState(dr_seed),
            dr_config_type="custom",
            dr_seed=dr_seed,
        )
        env.apply_startup_domain_rand_bundle(patch)
    return env


def evaluate_policy(
    policy_dir: Path,
    dr_cfg: DomainRandomizationConfig,
    seed: int,
) -> dict[str, float]:
    base_env = make_env_from_cfg(dr_cfg, dr_seed=1_000_000 + seed)
    vec = DummyVecEnv([lambda: base_env])
    vec = VecNormalize.load(str(policy_dir / "vec_normalize.pkl"), vec)
    vec.training = False
    vec.norm_reward = False
    model = SAC.load(str(policy_dir / "final_model.zip"), env=vec)

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
        for scenario_name, cfg in SCENARIOS.items():
            results = [evaluate_policy(policy_dir, cfg, seed) for seed in EVAL_SEEDS]
            print(
                scenario_name,
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
