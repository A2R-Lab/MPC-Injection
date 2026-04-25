from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from mpc_rl.envs.domain_randomization import resolve_startup_domain_rand_config
from mpc_rl.envs.velocity_tracking_env import QuadrupedVelocityTrackingEnv


POLICIES = {
    "old_success_256_s100": Path(
        "logs/quadruped_domain_rand_mpc_dr_new_data_test/"
        "SAC-MPC-default_no_push/"
        "quadruped-velocity_tracking-SAC-MPC-20260417-134426-"
        "percentage-25pct-seed100-env256-drdefault_no_push"
    ),
    "sysid_256_s100": Path(
        "logs/quadruped_domain_rand_mpc_sys_id/"
        "SAC-MPC-sysid_floor_sensing_no_push/"
        "quadruped-velocity_tracking-SAC-MPC-20260423-195108-"
        "percentage-25pct-seed100-env256-drsysid_floor_sensing_no_push"
    ),
    "sysid_512_s150": Path(
        "logs/quadruped_domain_rand_mpc_sys_id/"
        "SAC-MPC-sysid_floor_sensing_no_push/"
        "quadruped-velocity_tracking-SAC-MPC-20260424-001153-"
        "percentage-25pct-seed150-env512-drsysid_floor_sensing_no_push"
    ),
    "sysid_32_s100": Path(
        "logs/quadruped_domain_rand_mpc_sys_id/"
        "SAC-MPC-sysid_floor_sensing_no_push/"
        "quadruped-velocity_tracking-SAC-MPC-20260424-115002-"
        "percentage-25pct-seed100-env32-drsysid_floor_sensing_no_push"
    ),
}

SCENARIOS = [
    ("nominal_clean", "disabled"),
    ("sysid_floor_only", "sysid_floor_only_no_push"),
    ("sysid_floor_sensing", "sysid_floor_sensing_no_push"),
    ("old_default_dr", "default_no_push"),
]

EVAL_SEEDS = [11, 22, 33]
COMMANDS = [0.5, 1.0]
HORIZON = 400


def make_env_from_scenario(preset_name: str, dr_seed: int) -> QuadrupedVelocityTrackingEnv:
    resolved_type, cfg = resolve_startup_domain_rand_config(preset_name)
    env = QuadrupedVelocityTrackingEnv(
        robot="go2",
        scene="flat",
        domain_rand_cfg=cfg,
        apply_startup_domain_rand_on_init=False,
        simple_reward=True,
    )
    if cfg.enable:
        patch = env.sample_startup_domain_rand_bundle(
            rng=np.random.RandomState(dr_seed),
            dr_config_type=resolved_type,
            dr_seed=dr_seed,
        )
        env.apply_startup_domain_rand_bundle(patch)
    return env


def evaluate_policy(policy_dir: Path, preset_name: str, seed: int, vx: float) -> dict[str, float]:
    base_env = make_env_from_scenario(preset_name, dr_seed=1_000_000 + seed)
    vec = DummyVecEnv([lambda: base_env])
    vec = VecNormalize.load(str(policy_dir / "vec_normalize.pkl"), vec)
    vec.training = False
    vec.norm_reward = False
    model = SAC.load(str(policy_dir / "final_model.zip"), env=vec)

    obs = vec.reset()
    base_env.set_commands(vx=vx, vy=0.0, wz=0.0)
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
        for scenario_name, preset in SCENARIOS:
            for vx in COMMANDS:
                results = [
                    evaluate_policy(policy_dir, preset, seed, vx) for seed in EVAL_SEEDS
                ]
                print(
                    scenario_name,
                    "vx",
                    vx,
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
