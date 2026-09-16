#!/bin/bash

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd -- "${REPO_ROOT}"

# Bash script to run a quadruped TD3-MPC diagnostic sweep with varying MPC
# injection percentages.
#
# This version is intentionally configured as a simpler ablation than the
# default sim2real setup:
# - Fewer parallel envs to increase UNIQUE MPC trajectory residency in replay
# - Domain randomization disabled to remove a major train/demo mismatch
#
# The goal is to answer a narrow question first:
#   "Can quadruped TD3-MPC learn the intended gait from MPC data in the
#    easier no-DR setting when replay is less diversity-starved?"

# Common parameters (edit these as needed)
ENV_NAME="quadruped-velocity_tracking"
ALGORITHM="SAC-MPC"
TOTAL_TIMESTEPS=1000000
NUM_ENVS=4
LEARNING_STARTS=50000
SAVE_REPLAY_BUFFER_CHECKPOINTS="False"
SAVE_REPLAY_BUFFER_FINAL="True"
DOMAIN_RAND="False"
DOMAIN_RAND_OBS_NOISE=0.0
USE_GO2_SYSID="True"
DATA_DIR="data/quadruped/"
#DATA_DIR="data/quadruped_dr/sysid_nominal/" # Misleading dir placement, doesn't have DR, just sys ID data
QUADRUPED_MPC_REPLAY_MODE="torque_current_pd" # direct (used when data has converted pd) | torque_saved_pd | torque_current_pd (used for re-calculating pd)
BUFFER_SIZE=5000000
LEARNING_RATE=3e-4 # OG 3e-4
POLICY_DELAY=2
BATCH_SIZE=256 # 256 for sim, 512 for sim2real
LOG_DIR="logs/SAC-MPC-quadruped_x_vel_only_runs_min_rwrd_ablation/"
#LOG_DIR="logs/SAC-MPC-quadruped_x_vel_only_runs_min_rwrd_tiles_256/"
#LOG_DIR="logs/SAC-MPC-quadruped_sysid_nominal/"

# MPC percentage sweep values
PERCENTAGES=(0 25 50 75 100)

# Seeds to sweep over
#SEEDS=(100 150)
#SEEDS=(200 250)
SEEDS=(300 350)

# Additional checkpoint video evaluations to record during each run.
# Bash arrays are space-separated; the script converts them to the
# comma-separated --checkpoint_evals flag expected by train.py.
CHECKPOINT_EVALS=(500000)
CHECKPOINT_EVALS_CSV=$(IFS=,; echo "${CHECKPOINT_EVALS[*]}")

echo "Starting ${ALGORITHM} quadruped percentage sweep experiments"
echo "==========================================================="
echo "Environment: ${ENV_NAME}"
echo "Algorithm: ${ALGORITHM}"
echo "Total timesteps: ${TOTAL_TIMESTEPS}"
echo "Num envs: ${NUM_ENVS}"
echo "Seeds: ${SEEDS[*]}"
echo "Learning starts: ${LEARNING_STARTS}"
echo "Save replay buffer checkpoints: ${SAVE_REPLAY_BUFFER_CHECKPOINTS}"
echo "Save replay buffer final: ${SAVE_REPLAY_BUFFER_FINAL}"
echo "Domain randomization: ${DOMAIN_RAND}"
echo "Domain randomization obs noise: ${DOMAIN_RAND_OBS_NOISE}"
echo "Go2 sysID joint dynamics: ${USE_GO2_SYSID}"
echo "Data directory: ${DATA_DIR}"
echo "Buffer size: ${BUFFER_SIZE}"
echo "Learning rate: ${LEARNING_RATE}"
echo "Policy delay: ${POLICY_DELAY}"
echo "Batch size: ${BATCH_SIZE}"
echo "Log directory: ${LOG_DIR}"
echo "Percentages: ${PERCENTAGES[*]}"
echo "Checkpoint evals: ${CHECKPOINT_EVALS[*]}"
TOTAL_RUNS=$((${#SEEDS[@]} * ${#PERCENTAGES[@]}))
echo "Total runs: ${TOTAL_RUNS}"
echo "==========================================================="
echo ""

run_idx=0
for seed in "${SEEDS[@]}"; do
    for percentage in "${PERCENTAGES[@]}"; do
        run_idx=$((run_idx + 1))

        echo ""
        echo "=========================================="
        echo "Run ${run_idx}/${TOTAL_RUNS}: ${percentage}% MPC target, seed ${seed}"
        echo "=========================================="
        echo ""

        python mpc_rl/train.py \
            --env_name="${ENV_NAME}" \
            --algorithm="${ALGORITHM}" \
            --total_timesteps="${TOTAL_TIMESTEPS}" \
            --num_envs="${NUM_ENVS}" \
            --seed="${seed}" \
            --learning_starts="${LEARNING_STARTS}" \
            --save_replay_buffer_checkpoints="${SAVE_REPLAY_BUFFER_CHECKPOINTS}" \
            --save_replay_buffer_final="${SAVE_REPLAY_BUFFER_FINAL}" \
            --domain_rand="${DOMAIN_RAND}" \
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
            --suffix="seed${seed}"

        if [ $? -ne 0 ]; then
            echo ""
            echo "ERROR: Experiment with ${percentage}% MPC and seed ${seed} failed!"
            echo "Stopping experiment sweep."
            exit 1
        fi

        echo ""
        echo "Completed experiment with ${percentage}% MPC and seed ${seed}"
        echo ""
    done
done

echo ""
echo "=============================================="
echo "All experiments completed successfully!"
echo "=============================================="
