#!/bin/bash

set -euo pipefail

# Bash script to run TD3-MPC experiments for the three-legged cheetah task with
# varying MPC injection percentages.
#
# This mirrors run_walker_experiments.sh, but uses the local cheetah3
# environment and cheetah3 MPC trajectory directory.

# Common parameters
ENV_NAME="cheetah3-run"

ALGORITHM="SAC-MPC"
#ALGORITHM="TD3-MPC"

TOTAL_TIMESTEPS=500000
INJECT_TYPE="percentage"
RANDOM_SELECT="True"
SAVE_REPLAY_BUFFER_CHECKPOINTS="False"
SAVE_REPLAY_BUFFER_FINAL="False"

DATA_DIR="data/cheetah3_0_010dt/"
LOG_DIR="logs/SAC-MPC-cheetah3-runs/1st_run/"

CHECKPOINT_FREQ=25000
CHEETAH3_SPEED_GOAL=3.0

# Sweep dimensions
PERCENTAGES=(0 25 50 75 100)
SEEDS=(1 50 100 150 200)

echo "Starting ${ALGORITHM} cheetah3 percentage sweep experiments"
echo "==========================================================="
echo "Environment: ${ENV_NAME}"
echo "Algorithm: ${ALGORITHM} with TaggedReplayBuffer"
echo "Total timesteps: ${TOTAL_TIMESTEPS}"
echo "Injection type: ${INJECT_TYPE} (checks before each train())"
echo "Data directory: ${DATA_DIR}"
echo "Random selection: ${RANDOM_SELECT}"
echo "Save replay buffer checkpoints: ${SAVE_REPLAY_BUFFER_CHECKPOINTS}"
echo "Save replay buffer final: ${SAVE_REPLAY_BUFFER_FINAL}"
echo "Cheetah3 speed goal: ${CHEETAH3_SPEED_GOAL}"
echo "Log directory: ${LOG_DIR}"
echo "Percentages: ${PERCENTAGES[*]}"
echo "Seeds: ${SEEDS[*]}"
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

        if ! python mpc_rl/train.py \
            --env_name="${ENV_NAME}" \
            --algorithm="${ALGORITHM}" \
            --total_timesteps="${TOTAL_TIMESTEPS}" \
            --inject_type="${INJECT_TYPE}" \
            --percentage="${percentage}" \
            --random_select="${RANDOM_SELECT}" \
            --data_dir="${DATA_DIR}" \
            --logdir="${LOG_DIR}" \
            --seed="${seed}" \
            --save_replay_buffer_checkpoints="${SAVE_REPLAY_BUFFER_CHECKPOINTS}" \
            --save_replay_buffer_final="${SAVE_REPLAY_BUFFER_FINAL}" \
            --cheetah3_speed_goal="${CHEETAH3_SPEED_GOAL}" \
            --checkpoint_freq="${CHECKPOINT_FREQ}" \
            --suffix="seed${seed}"; then
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
