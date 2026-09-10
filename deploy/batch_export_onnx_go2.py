"""Batch-export Go2 MPC-Injection policies from training logs to deploy directories.

The script scans a log root containing many training run directories and exports
every available final model and requested checkpoint into the directory layout
expected by deploy/robots/go2/build/go2_ctrl:

    <policy_dir>/params/deploy.yaml
    <policy_dir>/exported/policy.onnx

It also writes a manifest and ready-to-run go2_ctrl command list so real-robot
policy sweeps do not require manually editing RUN variables.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_LOG_ROOT = Path(
    "logs/quadruped_domain_rand_mpc_dr_sysid_dyn20_mjlab_10k_LPF/"
    "SAC-MPC-sysid_dyn20_mjlab"
)
DEFAULT_POLICY_ROOT = Path("deploy/robots/go2/config/policy/velocity/policies")
DEFAULT_DEPLOY_PARAMS = Path(
    "deploy/robots/go2/config/policy/velocity/v0/params/deploy.yaml"
)
GO2_PROJECT_ROOT = Path("deploy/robots/go2")


@dataclass(frozen=True)
class ExportCandidate:
    policy_name: str
    kind: str
    run_dir: Path
    model_zip: Path
    vecnorm_pkl: Path
    policy_dir: Path


def repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def sanitize_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_.-")
    return value or "policy"


def short_run_name(run_name: str) -> str:
    timestamp_match = re.search(r"(\d{8}-\d{6})", run_name)
    seed_match = re.search(r"seed(\d+)", run_name)
    percent_match = re.search(r"percentage-([^-]+)", run_name)
    env_match = re.search(r"env(\d+)", run_name)

    parts = []
    if timestamp_match:
        parts.append(timestamp_match.group(1))
    if seed_match:
        parts.append(f"seed{seed_match.group(1)}")
    if percent_match:
        parts.append(percent_match.group(1))
    if env_match:
        parts.append(f"env{env_match.group(1)}")

    if parts:
        return sanitize_name("_".join(parts))
    return sanitize_name(run_name)[:120]


def candidate_needs_export(
    candidate: ExportCandidate,
    deploy_params: Path,
    *,
    force: bool,
) -> bool:
    if force:
        return True

    output_path = candidate.policy_dir / "exported" / "policy.onnx"
    params_path = candidate.policy_dir / "params" / "deploy.yaml"
    if not output_path.exists() or not params_path.exists():
        return True

    output_mtime = min(output_path.stat().st_mtime, params_path.stat().st_mtime)
    sources = [candidate.model_zip, candidate.vecnorm_pkl, deploy_params]
    return any(source.stat().st_mtime > output_mtime for source in sources)


def discover_candidates(
    log_root: Path,
    policy_root: Path,
    *,
    checkpoint_steps: list[int],
    include_final: bool,
    include_checkpoints: bool,
) -> tuple[list[ExportCandidate], list[str]]:
    candidates: list[ExportCandidate] = []
    skipped: list[str] = []
    used_names: set[str] = set()

    for run_dir in sorted(path for path in log_root.iterdir() if path.is_dir()):
        run_short = short_run_name(run_dir.name)

        def add_candidate(kind: str, model_zip: Path, vecnorm_pkl: Path) -> None:
            if model_zip.exists() and vecnorm_pkl.exists():
                policy_name = sanitize_name(f"{run_short}_{kind}")
                unique_name = policy_name
                suffix = 2
                while unique_name in used_names:
                    unique_name = f"{policy_name}_{suffix}"
                    suffix += 1
                used_names.add(unique_name)
                candidates.append(
                    ExportCandidate(
                        policy_name=unique_name,
                        kind=kind,
                        run_dir=run_dir,
                        model_zip=model_zip,
                        vecnorm_pkl=vecnorm_pkl,
                        policy_dir=policy_root / unique_name,
                    )
                )
                return

            missing = []
            if not model_zip.exists():
                missing.append(display_path(model_zip))
            if not vecnorm_pkl.exists():
                missing.append(display_path(vecnorm_pkl))
            skipped.append(f"{run_dir.name} {kind}: missing {', '.join(missing)}")

        if include_final:
            add_candidate(
                "final",
                run_dir / "final_model.zip",
                run_dir / "vec_normalize.pkl",
            )

        if include_checkpoints:
            for step in checkpoint_steps:
                add_candidate(
                    f"{step // 1000}k" if step % 1000 == 0 else str(step),
                    run_dir / "checkpoints" / f"model_{step}_steps.zip",
                    run_dir / "checkpoints" / f"model_vecnormalize_{step}_steps.pkl",
                )

    return candidates, skipped


def copy_deploy_params(deploy_params: Path, policy_dir: Path) -> None:
    params_dir = policy_dir / "params"
    params_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(deploy_params, params_dir / "deploy.yaml")


def go2_policy_dir_arg(policy_dir: Path) -> str:
    go2_root = repo_path(GO2_PROJECT_ROOT).resolve()
    resolved_policy_dir = policy_dir.resolve()
    try:
        return resolved_policy_dir.relative_to(go2_root).as_posix()
    except ValueError:
        return str(resolved_policy_dir)


def launch_command(candidate: ExportCandidate, network: str) -> str:
    return (
        f"./go2_ctrl --network={network} "
        f"--policy_dir={go2_policy_dir_arg(candidate.policy_dir)}"
    )


def write_outputs(
    policy_root: Path,
    rows: list[dict[str, str]],
    commands: list[str],
) -> tuple[Path, Path]:
    policy_root.mkdir(parents=True, exist_ok=True)
    manifest_path = policy_root / "manifest.tsv"
    commands_path = policy_root / "launch_commands.txt"

    fieldnames = [
        "policy_name",
        "kind",
        "status",
        "run_dir",
        "model_zip",
        "vecnorm_pkl",
        "policy_dir",
        "go2_policy_dir_arg",
        "launch_command",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest:
        writer = csv.DictWriter(manifest, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    with commands_path.open("w", encoding="utf-8") as command_file:
        command_file.write("# Run from deploy/robots/go2/build\n")
        for command in commands:
            command_file.write(command)
            command_file.write("\n")

    return manifest_path, commands_path


def run_export(
    candidate: ExportCandidate,
    *,
    deploy_params: Path,
    algo: str | None,
    opset: int,
) -> None:
    from export_onnx_go2 import export

    copy_deploy_params(deploy_params, candidate.policy_dir)
    export(
        model_zip=candidate.model_zip,
        vecnorm_pkl=candidate.vecnorm_pkl,
        output_dir=candidate.policy_dir / "exported",
        algo=algo,
        opset_version=opset,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-export Go2 ONNX policies from a directory of training logs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--log_root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--policy_root", type=Path, default=DEFAULT_POLICY_ROOT)
    parser.add_argument("--deploy_params", type=Path, default=DEFAULT_DEPLOY_PARAMS)
    parser.add_argument(
        "--checkpoint_step",
        type=int,
        action="append",
        default=None,
        help="Checkpoint step to export. Repeat for multiple checkpoints.",
    )
    parser.add_argument("--skip_final", action="store_true")
    parser.add_argument("--skip_checkpoints", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-export even if outputs are current.")
    parser.add_argument("--dry_run", action="store_true", help="Print planned exports without writing files.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of candidates, useful for smoke tests.")
    parser.add_argument("--algo", choices=["SAC", "TD3"], default=None)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--network", default="network_name", help="Network name used in generated launch commands.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_root = repo_path(args.log_root).resolve()
    policy_root = repo_path(args.policy_root).resolve()
    deploy_params = repo_path(args.deploy_params).resolve()
    checkpoint_steps = args.checkpoint_step or [900000]

    if not log_root.exists():
        raise SystemExit(f"log_root not found: {display_path(log_root)}")
    if not deploy_params.exists():
        raise SystemExit(f"deploy_params not found: {display_path(deploy_params)}")

    candidates, skipped = discover_candidates(
        log_root,
        policy_root,
        checkpoint_steps=checkpoint_steps,
        include_final=not args.skip_final,
        include_checkpoints=not args.skip_checkpoints,
    )
    if args.limit is not None:
        candidates = candidates[: args.limit]

    rows: list[dict[str, str]] = []
    commands: list[str] = []
    exported = 0
    current = 0

    print(f"Log root:     {display_path(log_root)}")
    print(f"Policy root:  {display_path(policy_root)}")
    print(f"Candidates:   {len(candidates)}")
    if skipped:
        print(f"Skipped incomplete entries: {len(skipped)}")

    for index, candidate in enumerate(candidates, start=1):
        needs_export = candidate_needs_export(
            candidate,
            deploy_params,
            force=args.force,
        )
        status = "dry-run" if args.dry_run else ("exported" if needs_export else "current")
        command = launch_command(candidate, args.network)

        print(
            f"[{index:03d}/{len(candidates):03d}] {candidate.policy_name}: "
            f"{status}"
        )
        if args.dry_run:
            print(f"    {command}")
        elif needs_export:
            run_export(
                candidate,
                deploy_params=deploy_params,
                algo=args.algo,
                opset=args.opset,
            )
            exported += 1
        else:
            current += 1

        rows.append(
            {
                "policy_name": candidate.policy_name,
                "kind": candidate.kind,
                "status": status,
                "run_dir": display_path(candidate.run_dir),
                "model_zip": display_path(candidate.model_zip),
                "vecnorm_pkl": display_path(candidate.vecnorm_pkl),
                "policy_dir": display_path(candidate.policy_dir),
                "go2_policy_dir_arg": go2_policy_dir_arg(candidate.policy_dir),
                "launch_command": command,
            }
        )
        commands.append(command)

    if not args.dry_run:
        manifest_path, commands_path = write_outputs(policy_root, rows, commands)
        print(f"\nExported: {exported}")
        print(f"Already current: {current}")
        print(f"Manifest: {display_path(manifest_path)}")
        print(f"Launch commands: {display_path(commands_path)}")
    else:
        print("\nDry run only; no files were written.")


if __name__ == "__main__":
    main()
