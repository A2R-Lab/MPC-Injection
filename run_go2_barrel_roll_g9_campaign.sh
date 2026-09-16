#!/usr/bin/env bash

set -euo pipefail

readonly REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly DATA_DIR="data/go2_barrel_roll/v3"
readonly DATA_PATH="${REPO_ROOT}/${DATA_DIR}"
readonly EXPECTED_CHECKSUM_INDEX_SHA256="3c4401e353b3195e7c8801ae29257e84598f099c3bfab05b37dcb1c80c62c2cc"
readonly EXPECTED_AGGREGATE_MANIFEST_SHA256="85eb1703d145f814f23f430d0d16f9e8697f99d61806917b454efe0c380d11d9"
readonly EXPECTED_EFFECTIVE_CONFIG_SHA256="65a7727010fa9a1052894ac0d74ced21e827a5ad84be978756724f0ff597a548"
readonly -a SEEDS=(1 2)

mode="${1:-production}"
if [[ "${mode}" != "smoke" && "${mode}" != "production" ]]; then
    echo "Usage: $0 [smoke|production]" >&2
    exit 2
fi
if [[ "${mode}" == "smoke" ]]; then
    readonly TOTAL_TIMESTEPS=10000
    readonly CAMPAIGN_DIR="${REPO_ROOT}/logs/go2_barrel_roll_g9/resource_smoke"
    readonly SUFFIX="go2-barrel-roll-g9-resource-smoke"
else
    readonly TOTAL_TIMESTEPS=500000
    readonly CAMPAIGN_DIR="${REPO_ROOT}/logs/go2_barrel_roll_g9_production"
    readonly SUFFIX="go2-barrel-roll-v3"
fi

job_pids=()
monitor_pid=""

terminate_jobs() {
    local pid
    for pid in "${job_pids[@]:-}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            kill -TERM -- "-${pid}" 2>/dev/null || true
        fi
    done
}

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n "${monitor_pid}" ]]; then
        kill "${monitor_pid}" 2>/dev/null || true
        wait "${monitor_pid}" 2>/dev/null || true
    fi
    if (( status != 0 )); then
        terminate_jobs
    fi
    exit "${status}"
}
trap 'exit 130' INT
trap 'exit 143' TERM
trap cleanup EXIT

cd -- "${REPO_ROOT}"
if [[ "${EXPECTED_CHECKSUM_INDEX_SHA256}" == REPLACE_* \
    || "${EXPECTED_AGGREGATE_MANIFEST_SHA256}" == REPLACE_* \
    || "${EXPECTED_EFFECTIVE_CONFIG_SHA256}" == REPLACE_* ]]; then
    echo "ERROR: hard-coded schema-v3 identities have not been frozen after promotion." >&2
    exit 1
fi
if [[ -e "${CAMPAIGN_DIR}" ]]; then
    echo "ERROR: refusing to overwrite G9 ${mode} campaign: ${CAMPAIGN_DIR}" >&2
    exit 1
fi
if [[ ! -d "${DATA_PATH}" ]]; then
    echo "ERROR: schema-v3 production dataset is missing: ${DATA_PATH}" >&2
    exit 1
fi

actual_checksum_hash="$(sha256sum "${DATA_PATH}/checksums.sha256" | awk '{print $1}')"
actual_aggregate_hash="$(sha256sum "${DATA_PATH}/generation_manifest_aggregate.jsonl" | awk '{print $1}')"
actual_config_hash="$(conda run --no-capture-output -n mpc-rl python - "${DATA_PATH}/dataset_summary.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["effective_config_sha256"])
PY
)"
if [[ "${actual_checksum_hash}" != "${EXPECTED_CHECKSUM_INDEX_SHA256}" ]]; then
    echo "ERROR: schema-v3 checksum-index identity changed: ${actual_checksum_hash}" >&2
    exit 1
fi
if [[ "${actual_aggregate_hash}" != "${EXPECTED_AGGREGATE_MANIFEST_SHA256}" ]]; then
    echo "ERROR: schema-v3 aggregate-manifest identity changed: ${actual_aggregate_hash}" >&2
    exit 1
fi
if [[ "${actual_config_hash}" != "${EXPECTED_EFFECTIVE_CONFIG_SHA256}" ]]; then
    echo "ERROR: schema-v3 effective-configuration identity changed: ${actual_config_hash}" >&2
    exit 1
fi

mkdir -p -- "${CAMPAIGN_DIR}/runs"
conda run --no-capture-output -n mpc-rl python \
    mpc_rl/planner/barrel_roll_dataset.py "${DATA_PATH}" \
    >"${CAMPAIGN_DIR}/dataset_validation.log" 2>&1
(
    cd -- "${DATA_PATH}"
    sha256sum --check --strict checksums.sha256
) >"${CAMPAIGN_DIR}/checksum_validation.log" 2>&1

read -r baseline_swap_total_kib baseline_swap_free_kib < <(
    awk '
        /SwapTotal:/ {total=$2}
        /SwapFree:/ {free=$2}
        END {print total, free}
    ' /proc/meminfo
)
readonly BASELINE_SWAP_USED_KIB=$((baseline_swap_total_kib - baseline_swap_free_kib))

{
    echo "mode=${mode}"
    echo "root_commit=$(git rev-parse HEAD)"
    echo "root_tracked_status_sha256=$(git status --porcelain --untracked-files=no | sha256sum | awk '{print $1}')"
    echo "root_tracked_diff_sha256=$(git diff --binary -- . ':(exclude)deploy/robots/go2/config/policy/velocity/policies/**' | sha256sum | awk '{print $1}')"
    echo "mpx_commit=$(git -C deps/mpx rev-parse HEAD)"
    echo "mpx_tracked_status_sha256=$(git -C deps/mpx status --porcelain --untracked-files=no | sha256sum | awk '{print $1}')"
    echo "mpx_tracked_diff_sha256=$(git -C deps/mpx diff --binary | sha256sum | awk '{print $1}')"
    echo "solver_commit=$(git -C deps/mpx/mpx/primal_dual_ilqr rev-parse HEAD)"
    echo "gym_quadruped_commit=$(git -C deps/gym-quadruped rev-parse HEAD)"
    echo "mujoco_mpc_commit=$(git -C deps/mujoco_mpc rev-parse HEAD)"
    echo "checksum_index_sha256=${actual_checksum_hash}"
    echo "aggregate_manifest_sha256=${actual_aggregate_hash}"
    echo "effective_config_sha256=${actual_config_hash}"
    echo "seeds=${SEEDS[*]}"
    echo "total_timesteps=${TOTAL_TIMESTEPS}"
    echo "num_envs=4"
    echo "target_mpc_percentage=25"
    echo "baseline_swap_used_kib=${BASELINE_SWAP_USED_KIB}"
} >"${CAMPAIGN_DIR}/campaign_provenance.txt"

for seed in "${SEEDS[@]}"; do
    mkdir -p -- "${CAMPAIGN_DIR}/runs/seed${seed}"
    setsid bash -c 'exec "$@"' bash \
        env PYTHONUNBUFFERED=1 \
        /usr/bin/time -v -o "${CAMPAIGN_DIR}/seed${seed}_time.txt" \
        conda run --no-capture-output -n mpc-rl python mpc_rl/train.py \
        --env_name=quadruped-barrel_roll \
        --robot=go2 \
        --algorithm=SAC-MPC \
        --total_timesteps="${TOTAL_TIMESTEPS}" \
        --num_envs=4 \
        --max_episode_steps=125 \
        --learning_rate=0.0003 \
        --buffer_size=1000000 \
        --learning_starts=10000 \
        --batch_size=256 \
        --tau=0.005 \
        --gamma=0.99 \
        --gradient_steps=-1 \
        --inject_type=percentage \
        --percentage=25 \
        --random_select=True \
        --quadruped_mpc_replay_mode=direct \
        --data_dir="${DATA_DIR}" \
        --domain_rand=False \
        --domain_rand_config_type=disabled \
        --use_go2_sysid=True \
        --checkpoint_freq=25000 \
        --eval_freq=10000 \
        --save_replay_buffer_checkpoints=False \
        --save_replay_buffer_final=False \
        --enable_logging=True \
        --logdir="${CAMPAIGN_DIR}/runs/seed${seed}" \
        --seed="${seed}" \
        --suffix="${SUFFIX}" \
        >"${CAMPAIGN_DIR}/seed${seed}_stdout.log" 2>&1 &
    job_pids+=("$!")
done

(
    while :; do
        alive=0
        for pid in "${job_pids[@]}"; do
            if kill -0 "${pid}" 2>/dev/null; then
                alive=$((alive + 1))
            fi
        done
        (( alive > 0 )) || break
        gpu_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
            | awk '{sum += $1} END {print sum + 0}')"
        read -r mem_available_kib swap_total_kib swap_free_kib < <(
            awk '
                /MemAvailable:/ {available=$2}
                /SwapTotal:/ {total=$2}
                /SwapFree:/ {free=$2}
                END {print available, total, free}
            ' /proc/meminfo
        )
        session_ids="$(IFS=,; echo "${job_pids[*]}")"
        total_cpu="$(ps -eo sid=,%cpu= 2>/dev/null | awk -v ids="${session_ids}" '
            BEGIN {count=split(ids, values, ","); for (i=1; i<=count; i++) wanted[values[i]]=1}
            $1 in wanted {sum += $2}
            END {print sum + 0}
        ')"
        printf '{"timestamp":"%s","processes_alive":%s,"gpu_memory_used_mib":%s,"host_mem_available_kib":%s,"swap_used_kib":%s,"training_cpu_percent":%s}\n' \
            "$(date --utc +%FT%TZ)" "${alive}" "${gpu_mib}" "${mem_available_kib}" \
            "$((swap_total_kib - swap_free_kib))" "${total_cpu}" \
            >>"${CAMPAIGN_DIR}/resource_samples.jsonl"
        sleep 1
    done
) &
monitor_pid="$!"

job_failed=0
active_pids=("${job_pids[@]}")
while (( ${#active_pids[@]} > 0 )); do
    completed_pid=""
    if wait -n -p completed_pid "${active_pids[@]}"; then
        next_active=()
        for pid in "${active_pids[@]}"; do
            [[ "${pid}" == "${completed_pid}" ]] || next_active+=("${pid}")
        done
        active_pids=("${next_active[@]}")
    else
        job_failed=1
        terminate_jobs
        break
    fi
done
if (( job_failed != 0 )); then
    wait "${job_pids[@]}" 2>/dev/null || true
    echo "ERROR: one training process failed; the other was terminated and logs were preserved." >&2
    exit 1
fi
wait "${job_pids[@]}" 2>/dev/null || true
kill "${monitor_pid}" 2>/dev/null || true
wait "${monitor_pid}" 2>/dev/null || true
monitor_pid=""

if rg -i -n 'out of memory|oom-kill|cuda error|killed process' \
    "${CAMPAIGN_DIR}"/seed*_stdout.log >"${CAMPAIGN_DIR}/oom_scan.txt"; then
    echo "ERROR: OOM/CUDA failure text found in training logs." >&2
    exit 1
fi

conda run --no-capture-output -n mpc-rl python - \
    "${CAMPAIGN_DIR}" "${mode}" "${TOTAL_TIMESTEPS}" "${BASELINE_SWAP_USED_KIB}" <<'PY' \
    >"${CAMPAIGN_DIR}/campaign_validation.json"
import json
import math
import sys
from pathlib import Path

campaign = Path(sys.argv[1])
mode = sys.argv[2]
total_timesteps = int(sys.argv[3])
baseline_swap = int(sys.argv[4])
expected_steps = list(range(0, total_timesteps + 1, 10_000))
run_reports = []
for seed in (1, 2):
    seed_root = campaign / "runs" / f"seed{seed}"
    configs = list(seed_root.glob("*/config.json"))
    if len(configs) != 1:
        raise SystemExit(f"seed {seed}: expected one run, found {len(configs)}")
    run_dir = configs[0].parent
    config = json.loads(configs[0].read_text())
    expected_options = {
        "algorithm": "SAC-MPC",
        "learning_rate": 0.0003,
        "buffer_size": 1_000_000,
        "learning_starts": 10_000,
        "batch_size": 256,
        "tau": 0.005,
        "gamma": 0.99,
        "gradient_steps": -1,
        "policy_delay": 2,
        "seed": seed,
        "inject_n_timesteps": 5_000,
        "inject_type": "percentage",
        "percentage": 25,
        "num_traj": 10,
        "random_select": True,
        "data_dir": "data/go2_barrel_roll/v3",
        "quadruped_mpc_replay_mode": "direct",
        "use_go2_sysid": True,
        "env_name": "quadruped-barrel_roll",
        "total_timesteps": total_timesteps,
        "num_envs": 4,
        "max_episode_steps": 125,
        "checkpoint_freq": 25_000,
        "eval_freq": 10_000,
        "enable_logging": True,
        "save_replay_buffer_checkpoints": False,
        "save_replay_buffer_final": False,
    }
    mismatches = {
        name: {"expected": expected, "actual": config.get(name)}
        for name, expected in expected_options.items()
        if config.get(name) != expected
    }
    domain_randomization = config.get("domain_randomization", {})
    if not isinstance(domain_randomization, dict) or domain_randomization.get("enabled") is not False:
        mismatches["domain_randomization.enabled"] = {
            "expected": False,
            "actual": domain_randomization,
        }
    if mismatches:
        raise SystemExit(f"seed {seed}: frozen config mismatch: {mismatches}")
    diagnostics_path = run_dir / "barrel_roll_pilot_diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text())
    if diagnostics.get("status") != "complete":
        raise SystemExit(f"seed {seed}: diagnostics status is not complete")
    if diagnostics.get("q_batches_checked", 0) <= 0:
        raise SystemExit(f"seed {seed}: no critic diagnostics were checked")
    for name, bounds in diagnostics.get("ranges", {}).items():
        if len(bounds) != 2 or not all(math.isfinite(float(value)) for value in bounds):
            raise SystemExit(f"seed {seed}: non-finite diagnostic range {name}")
    replay_percentage = float(diagnostics.get("replay_percentage_last", math.nan))
    if not math.isfinite(replay_percentage) or abs(replay_percentage - 25.0) > 0.1:
        raise SystemExit(
            f"seed {seed}: final MPC replay percentage {replay_percentage} is not 25%"
        )
    history = [json.loads(line) for line in (run_dir / "barrel_roll_eval_history.jsonl").read_text().splitlines()]
    if [int(record["timesteps"]) for record in history] != expected_steps:
        raise SystemExit(f"seed {seed}: validation schedule mismatch")
    if any(len(record.get("episodes", [])) != 100 for record in history):
        raise SystemExit(f"seed {seed}: validation did not use 100 episodes")
    if not (run_dir / "final_model.zip").is_file() or not (run_dir / "vec_normalize.pkl").is_file():
        raise SystemExit(f"seed {seed}: final model artifacts are missing")
    if list(run_dir.rglob("*replay_buffer*")):
        raise SystemExit(f"seed {seed}: replay-buffer artifacts were saved")
    expected_checkpoints = list(range(25_000, total_timesteps + 1, 25_000))
    for step in expected_checkpoints:
        paths = (
            run_dir / "checkpoints" / f"model_{step}_steps.zip",
            run_dir / "checkpoints" / f"model_vecnormalize_{step}_steps.pkl",
        )
        if not all(path.is_file() for path in paths):
            raise SystemExit(f"seed {seed}: checkpoint {step} is incomplete")
    actual_models = sorted(
        int(path.name.removeprefix("model_").removesuffix("_steps.zip"))
        for path in (run_dir / "checkpoints").glob("model_[0-9]*_steps.zip")
    )
    actual_stats = sorted(
        int(path.name.removeprefix("model_vecnormalize_").removesuffix("_steps.pkl"))
        for path in (run_dir / "checkpoints").glob("model_vecnormalize_[0-9]*_steps.pkl")
    )
    if actual_models != expected_checkpoints or actual_stats != expected_checkpoints:
        raise SystemExit(f"seed {seed}: checkpoint inventory differs from the 25k schedule")
    run_reports.append({
        "seed": seed,
        "run_dir": str(run_dir),
        "replay_percentage_last": replay_percentage,
        "validation_count": len(history),
        "validation_best_success_rate": max(float(record["success_rate"]) for record in history),
        "diagnostic_ranges": diagnostics["ranges"],
    })

samples = [json.loads(line) for line in (campaign / "resource_samples.jsonl").read_text().splitlines()]
if not samples:
    raise SystemExit("resource monitor produced no samples")
overlap_samples = [sample for sample in samples if int(sample["processes_alive"]) == 2]
if not overlap_samples:
    raise SystemExit("resource monitor did not capture concurrent process overlap")
peak_gpu = max(int(sample["gpu_memory_used_mib"]) for sample in overlap_samples)
min_available = min(int(sample["host_mem_available_kib"]) for sample in overlap_samples)
max_swap = max(int(sample["swap_used_kib"]) for sample in overlap_samples)
if peak_gpu >= 28 * 1024:
    raise SystemExit(f"combined GPU memory gate failed: {peak_gpu} MiB")
if min_available < 16 * 1024 * 1024:
    raise SystemExit(f"host available-memory gate failed: {min_available} KiB")
if max_swap > baseline_swap:
    raise SystemExit(f"new swap pressure detected: baseline={baseline_swap}, max={max_swap} KiB")
summary = {
    "mode": mode,
    "total_timesteps_per_seed": total_timesteps,
    "runs": run_reports,
    "resource_sample_count": len(samples),
    "overlap_resource_sample_count": len(overlap_samples),
    "peak_gpu_memory_used_mib": peak_gpu,
    "minimum_host_mem_available_kib": min_available,
    "baseline_swap_used_kib": baseline_swap,
    "maximum_swap_used_kib": max_swap,
}
print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
PY

printf 'complete\n' >"${CAMPAIGN_DIR}/COMPLETE"
echo "G9 ${mode} campaign completed and passed resource/diagnostic validation."

if [[ "${mode}" == "production" ]]; then
    conda run --no-capture-output -n mpc-rl python \
        mpc_rl/evaluate_go2_barrel_roll_g9.py \
        --campaign-dir="${CAMPAIGN_DIR}" \
        >"${CAMPAIGN_DIR}/final_evaluation_stdout.log" 2>&1
    echo "G9 untouched final evaluation completed."
fi
