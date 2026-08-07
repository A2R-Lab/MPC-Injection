#!/usr/bin/env bash

set -euo pipefail

readonly REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly DATA_DIR="${REPO_ROOT}/data/go2_barrel_roll/v2"
readonly CAMPAIGN_DIR="${REPO_ROOT}/logs/go2_barrel_roll_g8_production"
readonly EXPECTED_CHECKSUM_INDEX_SHA256="153334c544fec09e8aee4fa74223bde0f5b1c6b4f0018ce1e65328fbf0ccee70"
readonly EXPECTED_AGGREGATE_MANIFEST_SHA256="8d67b8ab296bc4bd56cffac69e34a44f2b8da922b203e0b0b9c1b4ad931b5491"
readonly -a SEEDS=(1 2 3)

cd -- "${REPO_ROOT}"

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "ERROR: tracked root worktree changes would invalidate G8 provenance." >&2
    exit 1
fi
if [[ -n "$(git -C deps/mpx status --porcelain --untracked-files=no)" ]]; then
    echo "ERROR: tracked MPX worktree changes would invalidate G8 provenance." >&2
    exit 1
fi
if [[ -e "${CAMPAIGN_DIR}" ]]; then
    echo "ERROR: refusing to overwrite existing G8 campaign: ${CAMPAIGN_DIR}" >&2
    exit 1
fi

mkdir -p -- "${CAMPAIGN_DIR}/runs"

actual_checksum_hash="$(sha256sum "${DATA_DIR}/checksums.sha256" | awk '{print $1}')"
actual_aggregate_hash="$(sha256sum "${DATA_DIR}/generation_manifest_aggregate.jsonl" | awk '{print $1}')"
if [[ "${actual_checksum_hash}" != "${EXPECTED_CHECKSUM_INDEX_SHA256}" ]]; then
    echo "ERROR: production checksum-index hash changed: ${actual_checksum_hash}" >&2
    exit 1
fi
if [[ "${actual_aggregate_hash}" != "${EXPECTED_AGGREGATE_MANIFEST_SHA256}" ]]; then
    echo "ERROR: production aggregate-manifest hash changed: ${actual_aggregate_hash}" >&2
    exit 1
fi

conda run --no-capture-output -n mpc-rl python \
    mpc_rl/planner/barrel_roll_dataset.py "${DATA_DIR}" \
    >"${CAMPAIGN_DIR}/dataset_validation.log" 2>&1
(
    cd -- "${DATA_DIR}"
    sha256sum --check --strict checksums.sha256
) >"${CAMPAIGN_DIR}/checksum_validation.log" 2>&1

{
    echo "root_commit=$(git rev-parse HEAD)"
    echo "mpx_commit=$(git -C deps/mpx rev-parse HEAD)"
    echo "solver_commit=$(git -C deps/mpx/mpx/primal_dual_ilqr rev-parse HEAD)"
    echo "gym_quadruped_commit=$(git -C deps/gym-quadruped rev-parse HEAD)"
    echo "mujoco_mpc_commit=$(git -C deps/mujoco_mpc rev-parse HEAD)"
    echo "checksum_index_sha256=${actual_checksum_hash}"
    echo "aggregate_manifest_sha256=${actual_aggregate_hash}"
    echo "seeds=${SEEDS[*]}"
    echo "total_timesteps=500000"
} >"${CAMPAIGN_DIR}/campaign_provenance.txt"

echo "Starting frozen Go2 barrel-roll G8 campaign"
echo "Seeds: ${SEEDS[*]}"
echo "Campaign: ${CAMPAIGN_DIR}"

for seed in "${SEEDS[@]}"; do
    echo "Starting production seed ${seed}"
    PYTHONUNBUFFERED=1 /usr/bin/time -v \
        -o "${CAMPAIGN_DIR}/seed${seed}_resources.txt" \
        conda run --no-capture-output -n mpc-rl python mpc_rl/train.py \
        --env_name=quadruped-barrel_roll \
        --robot=go2 \
        --algorithm=SAC-MPC \
        --total_timesteps=500000 \
        --num_envs=4 \
        --inject_type=percentage \
        --percentage=25 \
        --quadruped_mpc_replay_mode=direct \
        --data_dir="${DATA_DIR}" \
        --domain_rand=False \
        --domain_rand_config_type=disabled \
        --use_go2_sysid=True \
        --checkpoint_freq=25000 \
        --eval_freq=10000 \
        --save_replay_buffer_checkpoints=False \
        --save_replay_buffer_final=False \
        --logdir="${CAMPAIGN_DIR}/runs" \
        --seed="${seed}" \
        --suffix="go2-barrel-roll-v2-g8-seed${seed}" \
        2>&1 | tee "${CAMPAIGN_DIR}/seed${seed}_stdout.log"
    echo "Completed production seed ${seed}"
done

printf 'complete\n' >"${CAMPAIGN_DIR}/COMPLETE"
echo "All frozen Go2 barrel-roll G8 runs completed."
