#!/bin/bash

# Bash script to run SAC-MPC experiments with varying MPC injection percentages
# Runs experiments from 0% to 100% MPC data in increments of 5%
#
# This uses the TaggedReplayBuffer + train()-triggered injection architecture:
# - SAC_MPC.train() checks MPC percentage before sampling from replay buffer
# - If actual_mpc% < target%, automatically injects MPC trajectories
# - Transitions are tagged (0=RL, 1=MPC) for accurate percentage tracking
# - TensorBoard logs show real-time buffer composition

# Common parameters
#ENV_NAME="cartpole-swingup"
ENV_NAME="walker-walk"
ALGORITHM="SAC-MPC"
TOTAL_TIMESTEPS=500000
INJECT_TYPE="percentage"  # Use percentage-based injection (not fixed)
RANDOM_SELECT="True"
#DATA_DIR="data/cartpole_0_001dt/"
DATA_DIR="data/walker_0_0025dt/"
LOG_DIR="logs/SAC-MPC-walker-runs/6th_run/"
SEED=3000 # 1 is default for train_sbx.py

echo "Starting SAC-MPC percentage sweep experiments"
echo "=============================================="
echo "Environment: ${ENV_NAME}"
echo "Algorithm: ${ALGORITHM} with TaggedReplayBuffer"
echo "Total timesteps: ${TOTAL_TIMESTEPS}"
echo "Injection type: ${INJECT_TYPE} (checks before each train())"
echo "Data directory: ${DATA_DIR}"
echo "Random selection: ${RANDOM_SELECT}"
echo "=============================================="
echo ""

# Loop through percentages from 0 to 100 in increments of 25
for percentage in {0..100..25}; do
    echo ""
    echo "=========================================="
    echo "Running experiment with ${percentage}% MPC target"
    echo "=========================================="
    echo ""
    
    # Note: inject_n_timesteps is not used for percentage-based injection
    # Injection is triggered automatically by SAC_MPC.train() when MPC% drops below target
    python mpc_rl/train_sbx.py \
        --env_name="${ENV_NAME}" \
        --algorithm="${ALGORITHM}" \
        --total_timesteps="${TOTAL_TIMESTEPS}" \
        --inject_type="${INJECT_TYPE}" \
        --percentage="${percentage}" \
        --random_select="${RANDOM_SELECT}" \
        --data_dir="${DATA_DIR}" \
        --logdir="${LOG_DIR}" \
        --seed="${SEED}"
    
    # Check if the previous command succeeded
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


