#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly STAGING_DIR="${REPO_ROOT}/data/go2_barrel_roll/v3_staging"
readonly TARGET_DIR="${REPO_ROOT}/data/go2_barrel_roll/v3"
readonly AUDIT_DIR="${REPO_ROOT}/logs/go2_barrel_roll_g9/production_dataset"
readonly -a START_SEEDS=(4000000 4100000 4200000 4300000)
readonly ACCEPTED_PER_WORKER=250
readonly MAX_ATTEMPTS_PER_WORKER=2500

worker_pids=()
monitor_pid=""

terminate_workers() {
    local pid
    for pid in "${worker_pids[@]:-}"; do
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
        terminate_workers
    fi
    exit "${status}"
}
trap 'exit 130' INT
trap 'exit 143' TERM
trap cleanup EXIT

cd -- "${REPO_ROOT}"
if [[ -e "${STAGING_DIR}" || -e "${TARGET_DIR}" ]]; then
    echo "ERROR: refusing to overwrite schema-v3 staging or promoted data." >&2
    exit 1
fi
if [[ -e "${AUDIT_DIR}" ]]; then
    echo "ERROR: refusing to overwrite G9 production-data audit: ${AUDIT_DIR}" >&2
    exit 1
fi

mkdir -p -- "${STAGING_DIR}" "${AUDIT_DIR}"

{
    echo "root_commit=$(git rev-parse HEAD)"
    echo "root_tracked_diff_sha256=$(git diff --binary -- . ':(exclude)deploy/robots/go2/config/policy/velocity/policies/**' | sha256sum | awk '{print $1}')"
    echo "mpx_commit=$(git -C deps/mpx rev-parse HEAD)"
    echo "mpx_tracked_diff_sha256=$(git -C deps/mpx diff --binary | sha256sum | awk '{print $1}')"
    echo "solver_commit=$(git -C deps/mpx/mpx/primal_dual_ilqr rev-parse HEAD)"
    echo "gym_quadruped_commit=$(git -C deps/gym-quadruped rev-parse HEAD)"
    echo "mujoco_mpc_commit=$(git -C deps/mujoco_mpc rev-parse HEAD)"
    echo "start_seeds=${START_SEEDS[*]}"
    echo "accepted_per_worker=${ACCEPTED_PER_WORKER}"
    echo "max_attempts_per_worker=${MAX_ATTEMPTS_PER_WORKER}"
} >"${AUDIT_DIR}/provenance.txt"

for worker in 0 1 2 3; do
    seed="${START_SEEDS[worker]}"
    setsid bash -c 'exec "$@"' bash \
        env PYTHONUNBUFFERED=1 \
        XLA_PYTHON_CLIENT_PREALLOCATE=false \
        XLA_PYTHON_CLIENT_ALLOCATOR=platform \
        /usr/bin/time -v -o "${AUDIT_DIR}/worker${worker}_time.txt" \
        conda run --no-capture-output -n mpc-rl python \
        mpc_rl/planner/gen_traj_data_barrel_roll.py \
        --num-trajectories="${ACCEPTED_PER_WORKER}" \
        --start-seed="${seed}" \
        --max-attempts="${MAX_ATTEMPTS_PER_WORKER}" \
        --output-dir="${STAGING_DIR}" \
        --manifest-filename="generation_manifest_worker${worker}.jsonl" \
        >"${AUDIT_DIR}/worker${worker}_stdout.log" 2>&1 &
    worker_pids+=("$!")
done

(
    while :; do
        alive=0
        total_cpu="0"
        for pid in "${worker_pids[@]}"; do
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
        session_ids="$(IFS=,; echo "${worker_pids[*]}")"
        total_cpu="$(ps -eo sid=,%cpu= 2>/dev/null | awk -v ids="${session_ids}" '
            BEGIN {count=split(ids, values, ","); for (i=1; i<=count; i++) wanted[values[i]]=1}
            $1 in wanted {sum += $2}
            END {print sum + 0}
        ')"
        printf '{"timestamp":"%s","workers_alive":%s,"gpu_memory_used_mib":%s,"host_mem_available_kib":%s,"swap_used_kib":%s,"worker_cpu_percent":%s}\n' \
            "$(date --utc +%FT%TZ)" "${alive}" "${gpu_mib}" "${mem_available_kib}" \
            "$((swap_total_kib - swap_free_kib))" "${total_cpu}" \
            >>"${AUDIT_DIR}/resource_samples.jsonl"
        sleep 1
    done
) &
monitor_pid="$!"

worker_failed=0
active_pids=("${worker_pids[@]}")
while (( ${#active_pids[@]} > 0 )); do
    completed_pid=""
    if wait -n -p completed_pid "${active_pids[@]}"; then
        next_active=()
        for pid in "${active_pids[@]}"; do
            [[ "${pid}" == "${completed_pid}" ]] || next_active+=("${pid}")
        done
        active_pids=("${next_active[@]}")
    else
        worker_failed=1
        terminate_workers
        break
    fi
done
if (( worker_failed != 0 )); then
    wait "${worker_pids[@]}" 2>/dev/null || true
    echo "ERROR: a production-data worker failed; all logs and staging data were preserved." >&2
    exit 1
fi
wait "${worker_pids[@]}" 2>/dev/null || true
kill "${monitor_pid}" 2>/dev/null || true
wait "${monitor_pid}" 2>/dev/null || true
monitor_pid=""

conda run --no-capture-output -n mpc-rl python \
    mpc_rl/planner/barrel_roll_dataset.py --aggregate "${STAGING_DIR}" \
    >"${AUDIT_DIR}/aggregate_validation.log" 2>&1
(
    cd -- "${STAGING_DIR}"
    sha256sum --check --strict checksums.sha256
) >"${AUDIT_DIR}/checksum_validation.log" 2>&1

conda run --no-capture-output -n mpc-rl python - "${STAGING_DIR}" "${REPO_ROOT}" <<'PY' \
    >"${AUDIT_DIR}/promotion_validation.json"
import json
import sys
from pathlib import Path

directory = Path(sys.argv[1])
repo_root = Path(sys.argv[2])
summary = json.loads((directory / "dataset_summary.json").read_text())
expected_ranges = (
    range(4_000_000, 4_002_500),
    range(4_100_000, 4_102_500),
    range(4_200_000, 4_202_500),
    range(4_300_000, 4_302_500),
)
expected = {
    "schema_version": 3,
    "file_count": 1_000,
    "transition_count": 125_000,
    "accepted_count": 1_000,
}
for key, value in expected.items():
    if summary.get(key) != value:
        raise SystemExit(f"{key}: expected {value}, got {summary.get(key)}")
seeds = summary.get("seeds")
if (
    not isinstance(seeds, list)
    or len(seeds) != 1_000
    or len(set(seeds)) != 1_000
):
    raise SystemExit("production seeds are not exactly 1,000 unique values")
if not all(any(seed in values for values in expected_ranges) for seed in seeds):
    raise SystemExit("a production seed lies outside its worker range")
for worker, (start_seed, allowed) in enumerate(
    zip((4_000_000, 4_100_000, 4_200_000, 4_300_000), expected_ranges)
):
    manifest = directory / f"generation_manifest_worker{worker}.jsonl"
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    if not 250 <= len(records) <= 2_500:
        raise SystemExit(f"worker {worker} attempt count is outside [250, 2500]")
    if [int(record["seed"]) for record in records] != list(
        range(start_seed, start_seed + len(records))
    ):
        raise SystemExit(f"worker {worker} seeds are not consecutive from {start_seed}")
    accepted = [record for record in records if record.get("accepted")]
    if len(accepted) != 250:
        raise SystemExit(f"worker {worker} accepted {len(accepted)} instead of 250")
    if any(int(record["seed"]) not in allowed for record in records):
        raise SystemExit(f"worker {worker} attempted a seed outside its assigned range")
reserved = set(range(2_000_000, 2_000_100)) | set(range(3_000_000, 3_000_100))
if reserved.intersection(seeds):
    raise SystemExit("production seeds overlap validation or final-test seeds")
commissioning_seeds = set()
audit_root = repo_root / "logs" / "go2_barrel_roll_g9"
for manifest in audit_root.rglob("generation_manifest*.jsonl"):
    for line in manifest.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("seed") is not None:
                commissioning_seeds.add(int(record["seed"]))
overlap = sorted(commissioning_seeds.intersection(seeds))
if overlap:
    raise SystemExit(f"production seeds overlap commissioning seeds: {overlap[:10]}")
if summary.get("action_change_sanity_ratio") is None or summary["action_change_sanity_ratio"] > 0.10:
    raise SystemExit("production action-change sanity ratio exceeds 10%")
print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
PY

mv -- "${STAGING_DIR}" "${TARGET_DIR}"
cp -- "${TARGET_DIR}/dataset_summary.json" "${AUDIT_DIR}/dataset_summary.json"
sha256sum \
    "${TARGET_DIR}/checksums.sha256" \
    "${TARGET_DIR}/generation_manifest_aggregate.jsonl" \
    "${TARGET_DIR}/dataset_summary.json" \
    >"${AUDIT_DIR}/promoted_identity.sha256"
printf 'complete\n' >"${AUDIT_DIR}/COMPLETE"
echo "Promoted immutable schema-v3 dataset to ${TARGET_DIR}."
