#!/bin/bash

set -euo pipefail

# Main quadruped sim2real sweep entrypoint.
#
# Edit DOMAIN_RAND_CONFIG_TYPE to choose which quadruped DR preset to run:
#   - default_no_push : default DR values from domain_randomization.py, but no pushes
#   - half_no_push    : reduced-strength DR, no pushes
#   - quarter_no_push : quarter-strength DR, no pushes
#   - default         : full default DR, including pushes
#   - custom          : legacy mode; uses DOMAIN_RAND_OBS_NOISE override only
#   - disabled        : disables DR entirely
#
# Then run this script to sweep over the configured parallel environment counts.

# Common experiment parameters
ENV_NAME="quadruped-velocity_tracking"
ALGORITHM="SAC-MPC"
TOTAL_TIMESTEPS=1000000
LEARNING_STARTS=50000
SAVE_REPLAY_BUFFER_CHECKPOINTS="False"
SAVE_REPLAY_BUFFER_FINAL="True"
DOMAIN_RAND="True"
DOMAIN_RAND_CONFIG_TYPE="default_no_push" # "default_no_push", "half_no_push", or "quarter_no_push"
# Only used when DOMAIN_RAND_CONFIG_TYPE="custom".
DOMAIN_RAND_OBS_NOISE=1.0
DATA_DIR="data/quadruped_dr/default_no_push/"
BUFFER_SIZE=5000000
LEARNING_RATE=3e-4
POLICY_DELAY=2
BATCH_SIZE=256
LOG_DIR="logs/quadruped_domain_rand_mpc_dr/${ALGORITHM}-${DOMAIN_RAND_CONFIG_TYPE}/"

# Sweep dimensions
#NUM_ENVS_SWEEP=(4 8 16 32 64 128 256)
NUM_ENVS_SWEEP=(256 128 64 32 16 8 4)
PERCENTAGES=(25)
SEEDS=(1)

# Optional checkpoint videos during training
CHECKPOINT_EVALS=(300000 400000 500000 600000 700000 800000 900000)
CHECKPOINT_EVALS_CSV=$(IFS=,; echo "${CHECKPOINT_EVALS[*]}")

echo "Starting quadruped domain-randomization sweep"
echo "============================================"
echo "Environment: ${ENV_NAME}"
echo "Algorithm: ${ALGORITHM}"
echo "Total timesteps: ${TOTAL_TIMESTEPS}"
echo "Learning starts: ${LEARNING_STARTS}"
echo "Save replay buffer checkpoints: ${SAVE_REPLAY_BUFFER_CHECKPOINTS}"
echo "Save replay buffer final: ${SAVE_REPLAY_BUFFER_FINAL}"
echo "Domain randomization flag: ${DOMAIN_RAND}"
echo "Domain randomization config type: ${DOMAIN_RAND_CONFIG_TYPE}"
echo "Domain randomization obs noise override: ${DOMAIN_RAND_OBS_NOISE}"
echo "Data directory: ${DATA_DIR}"
echo "Buffer size: ${BUFFER_SIZE}"
echo "Learning rate: ${LEARNING_RATE}"
echo "Policy delay: ${POLICY_DELAY}"
echo "Batch size: ${BATCH_SIZE}"
echo "Log directory: ${LOG_DIR}"
echo "Num env sweep: ${NUM_ENVS_SWEEP[*]}"
echo "Percentages: ${PERCENTAGES[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Checkpoint evals: ${CHECKPOINT_EVALS[*]}"
TOTAL_RUNS=$((${#NUM_ENVS_SWEEP[@]} * ${#PERCENTAGES[@]} * ${#SEEDS[@]}))
echo "Total runs: ${TOTAL_RUNS}"
echo "============================================"
echo ""

run_idx=0
for seed in "${SEEDS[@]}"; do
    for num_envs in "${NUM_ENVS_SWEEP[@]}"; do
        for percentage in "${PERCENTAGES[@]}"; do
            run_idx=$((run_idx + 1))
            run_suffix="seed${seed}-env${num_envs}-dr${DOMAIN_RAND_CONFIG_TYPE}"

            echo ""
            echo "=============================================================="
            echo "Run ${run_idx}/${TOTAL_RUNS}: envs=${num_envs}, ${percentage}% MPC, seed=${seed}"
            echo "=============================================================="
            echo ""

            if ! python mpc_rl/train.py \
                --env_name="${ENV_NAME}" \
                --algorithm="${ALGORITHM}" \
                --total_timesteps="${TOTAL_TIMESTEPS}" \
                --num_envs="${num_envs}" \
                --seed="${seed}" \
                --learning_starts="${LEARNING_STARTS}" \
                --save_replay_buffer_checkpoints="${SAVE_REPLAY_BUFFER_CHECKPOINTS}" \
                --save_replay_buffer_final="${SAVE_REPLAY_BUFFER_FINAL}" \
                --domain_rand="${DOMAIN_RAND}" \
                --domain_rand_config_type="${DOMAIN_RAND_CONFIG_TYPE}" \
                --domain_rand_obs_noise="${DOMAIN_RAND_OBS_NOISE}" \
                --data_dir="${DATA_DIR}" \
                --buffer_size="${BUFFER_SIZE}" \
                --learning_rate="${LEARNING_RATE}" \
                --policy_delay="${POLICY_DELAY}" \
                --batch_size="${BATCH_SIZE}" \
                --logdir="${LOG_DIR}" \
                --percentage="${percentage}" \
                --checkpoint_evals="${CHECKPOINT_EVALS_CSV}" \
                --suffix="${run_suffix}"; then
                echo ""
                echo "ERROR: envs=${num_envs}, percentage=${percentage}, seed=${seed} failed."
                echo "Stopping sweep."
                exit 1
            fi

            echo ""
            echo "Completed envs=${num_envs}, percentage=${percentage}, seed=${seed}"
            echo ""
        done
    done
done

echo ""
echo "=============================================="
echo "All domain-randomization sweep runs completed!"
echo "=============================================="
