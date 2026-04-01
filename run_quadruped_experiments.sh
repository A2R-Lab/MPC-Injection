#!/bin/bash

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
ALGORITHM="TD3-MPC"
TOTAL_TIMESTEPS=2000000
NUM_ENVS=8
SEED=1
LEARNING_STARTS=50000
SAVE_REPLAY_BUFFER_CHECKPOINTS="False"
SAVE_REPLAY_BUFFER_FINAL="False"
DOMAIN_RAND="False"
DOMAIN_RAND_OBS_NOISE=0.0
DATA_DIR="data/quadruped/"
BUFFER_SIZE=10000000
LEARNING_RATE=3e-4
POLICY_DELAY=2
BATCH_SIZE=256
LOG_DIR="logs/TD3-MPC-quadruped_no_dr_8env_runs/"

# MPC percentage sweep values
PERCENTAGES=(0 25 50 75 100)

echo "Starting ${ALGORITHM} quadruped percentage sweep experiments"
echo "==========================================================="
echo "Environment: ${ENV_NAME}"
echo "Algorithm: ${ALGORITHM}"
echo "Total timesteps: ${TOTAL_TIMESTEPS}"
echo "Num envs: ${NUM_ENVS}"
echo "Seed: ${SEED}"
echo "Learning starts: ${LEARNING_STARTS}"
echo "Save replay buffer checkpoints: ${SAVE_REPLAY_BUFFER_CHECKPOINTS}"
echo "Save replay buffer final: ${SAVE_REPLAY_BUFFER_FINAL}"
echo "Domain randomization: ${DOMAIN_RAND}"
echo "Domain randomization obs noise: ${DOMAIN_RAND_OBS_NOISE}"
echo "Data directory: ${DATA_DIR}"
echo "Buffer size: ${BUFFER_SIZE}"
echo "Learning rate: ${LEARNING_RATE}"
echo "Policy delay: ${POLICY_DELAY}"
echo "Batch size: ${BATCH_SIZE}"
echo "Log directory: ${LOG_DIR}"
echo "Percentages: ${PERCENTAGES[*]}"
echo "==========================================================="
echo ""

for percentage in "${PERCENTAGES[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running experiment with ${percentage}% MPC target"
    echo "=========================================="
    echo ""

    python mpc_rl/train.py \
        --env_name="${ENV_NAME}" \
        --algorithm="${ALGORITHM}" \
        --total_timesteps="${TOTAL_TIMESTEPS}" \
        --num_envs="${NUM_ENVS}" \
        --seed="${SEED}" \
        --learning_starts="${LEARNING_STARTS}" \
        --save_replay_buffer_checkpoints="${SAVE_REPLAY_BUFFER_CHECKPOINTS}" \
        --save_replay_buffer_final="${SAVE_REPLAY_BUFFER_FINAL}" \
        --domain_rand="${DOMAIN_RAND}" \
        --domain_rand_obs_noise="${DOMAIN_RAND_OBS_NOISE}" \
        --data_dir="${DATA_DIR}" \
        --buffer_size="${BUFFER_SIZE}" \
        --learning_rate="${LEARNING_RATE}" \
        --policy_delay="${POLICY_DELAY}" \
        --batch_size="${BATCH_SIZE}" \
        --logdir="${LOG_DIR}" \
        --percentage="${percentage}"

    if [ $? -ne 0 ]; then
        echo ""
        echo "ERROR: Experiment with ${percentage}% MPC failed!"
        echo "Stopping experiment sweep."
        exit 1
    fi

    echo ""
    echo "Completed experiment with ${percentage}% MPC"
    echo ""
done

echo ""
echo "=============================================="
echo "All experiments completed successfully!"
echo "=============================================="
