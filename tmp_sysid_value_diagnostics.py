import mujoco

import mpx.config.config_go2 as go2_config
from mpc_rl.envs.go2_sysid import GO2_SYSID_IDENTIFIED_JOINT_DYNAMICS


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(go2_config.model_path))
    print(
        "joint,raw_armature,sysid_armature,arm_ratio,"
        "raw_frictionloss,sysid_frictionloss,fric_ratio,"
        "raw_damping,sysid_damping,damp_ratio"
    )
    for joint_name, sysid in GO2_SYSID_IDENTIFIED_JOINT_DYNAMICS.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        dof_index = int(model.jnt_dofadr[joint_id])
        raw_arm = float(model.dof_armature[dof_index])
        raw_fric = float(model.dof_frictionloss[dof_index])
        raw_damp = float(model.dof_damping[dof_index])
        arm_ratio = sysid["armature"] / raw_arm if raw_arm else float("inf")
        fric_ratio = sysid["frictionloss"] / raw_fric if raw_fric else float("inf")
        damp_ratio = sysid["damping"] / raw_damp if raw_damp else float("inf")
        print(
            joint_name,
            raw_arm,
            sysid["armature"],
            round(arm_ratio, 3),
            raw_fric,
            sysid["frictionloss"],
            round(fric_ratio, 3) if raw_fric else "inf",
            raw_damp,
            sysid["damping"],
            round(damp_ratio, 3),
            sep=",",
        )

    pairs = [("FL", "FR"), ("RL", "RR"), ("FL", "RL"), ("FR", "RR")]
    print("\npaired relative diffs")
    for left, right in pairs:
        print(f"\nPAIR {left} {right}")
        for joint_type in ("hip_joint", "thigh_joint", "calf_joint"):
            left_name = f"{left}_{joint_type}"
            right_name = f"{right}_{joint_type}"
            left_vals = GO2_SYSID_IDENTIFIED_JOINT_DYNAMICS[left_name]
            right_vals = GO2_SYSID_IDENTIFIED_JOINT_DYNAMICS[right_name]
            for field_name in ("armature", "frictionloss", "damping"):
                lhs = left_vals[field_name]
                rhs = right_vals[field_name]
                rel = abs(lhs - rhs) / max((abs(lhs) + abs(rhs)) / 2.0, 1e-12)
                print(
                    joint_type,
                    field_name,
                    round(lhs, 4),
                    round(rhs, 4),
                    "rel_diff",
                    round(rel, 3),
                )


if __name__ == "__main__":
    main()
