#!/bin/bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd -- "${REPO_ROOT}"

# Main quadruped sim2real sweep entrypoint.
#
# Edit DOMAIN_RAND_CONFIG_TYPE to choose which quadruped DR preset to run:
#   - disabled        : disables DR entirely
#   - sysid_floor_only_no_push    : floor/contact-surface friction only
#   - sysid_floor_sensing_no_push : floor friction plus RL sensing noise/bias
#   - sysid_dyn10_default_no_push : default-no-push DR plus per-joint +/-10% dynamics
#   - sysid_dyn20_mjlab           : MjLab/IsaacLab Go2 DR, pushes, and per-joint +/-20% dynamics
#   - sysid_dyn20_mjlab_no_push   : MjLab/IsaacLab Go2 DR and per-joint +/-20% dynamics, no pushes
#   - default_no_push : default DR values from domain_randomization.py, but no pushes
#   - half_no_push    : reduced-strength DR, no pushes
#   - quarter_no_push : quarter-strength DR, no pushes
#   - default         : full default DR, including pushes
#   - custom          : legacy mode; uses DOMAIN_RAND_OBS_NOISE override only
#
# Then run this script to sweep over the configured parallel environment counts.

# Common experiment parameters
ENV_NAME="quadruped-velocity_tracking"
ALGORITHM="TD3"
TOTAL_TIMESTEPS=2000000
LEARNING_STARTS=50000
SAVE_REPLAY_BUFFER_CHECKPOINTS="False"
SAVE_REPLAY_BUFFER_FINAL="False"
DOMAIN_RAND="True"
DOMAIN_RAND_CONFIG_TYPE="sysid_dyn20_mjlab" # sysID + DR + pushing
# Only used when DOMAIN_RAND_CONFIG_TYPE="custom".
DOMAIN_RAND_OBS_NOISE=1.0
USE_GO2_SYSID="True"
DATA_DIR="data/quadruped_dr/sysid_dyn20_mjlab_10k/" # sysid_dyn10_default_no_push_10k
BUFFER_SIZE=5000000
LEARNING_RATE=3e-4 # OG 3e-4, but found that it caused catastrophic forgetting as training went on
POLICY_DELAY=2
BATCH_SIZE=512 # Increasing batch size cuz of DR from 256
#LOG_DIR="logs/quadruped_domain_rand_mpc_dr_new_data_env_changes/${ALGORITHM}-${DOMAIN_RAND_CONFIG_TYPE}-high_term_cost/"
#LOG_DIR="logs/quadruped_domain_rand_mpc_dr_new_data_env_changes/${ALGORITHM}-${DOMAIN_RAND_CONFIG_TYPE}-low_max_pitch_roll/"
#LOG_DIR="logs/quadruped_domain_rand_mpc_dr_sysid_dyn20_mjlab_10k_LPF/${ALGORITHM}-${DOMAIN_RAND_CONFIG_TYPE}/"
#LOG_DIR="logs/quadruped_domain_rand_sac/${ALGORITHM}-${DOMAIN_RAND_CONFIG_TYPE}/"
LOG_DIR="logs/quadruped_td3_lpf/"

# Sweep dimensions
NUM_ENVS_SWEEP=(256)
PERCENTAGES=(0)
#SEEDS=(801 802 803 804 805)
SEEDS=(806 807 808 809 888)

# Optional checkpoint videos during training
#CHECKPOINT_EVALS=(100000 200000 300000 400000 500000 600000 700000 800000 900000)
CHECKPOINT_EVALS=(900000)
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
echo "Go2 sysID joint dynamics: ${USE_GO2_SYSID}"
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
                --use_go2_sysid="${USE_GO2_SYSID}" \
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
