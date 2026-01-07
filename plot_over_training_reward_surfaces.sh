#!/bin/bash

# Generate and plot reward surfaces for all checkpoints during training
# Complete pipeline: for each checkpoint, generate jobs -> run evaluations -> aggregate -> plot

# Activate conda environment
source ~/.bash_conda && conda activate mpc-rl

# =============================================================================
# Configuration - Modify these variables
# =============================================================================

# Training run directory containing checkpoints
#TRAINING_RUN_DIR="logs/SAC-MPC-walker-velocity_only_reward/3rd_run/walker-walk-SAC-MPC-20260107-112012-percentage-0pct"

TRAINING_RUN_DIR="logs/SAC-MPC-walker-velocity_only_reward/3rd_run/walker-walk-SAC-MPC-20260107-113507-percentage-50pct"

# Surface generation parameters
GRID_SIZE=31
MAGNITUDE=3
NUM_EPISODES=25
NUM_CPUS=16

# =============================================================================
# Parse directory name to extract metadata
# =============================================================================

# Extract percentage from directory name (e.g., "percentage-0pct" -> "0")
PERCENTAGE=$(echo "${TRAINING_RUN_DIR}" | grep -oP 'percentage-\K\d+(?=pct)')

# Extract algorithm name (e.g., "SAC-MPC")
ALGO=$(echo "${TRAINING_RUN_DIR}" | grep -oP 'walker-walk-\K[^-]+-[^-]+(?=-\d{8})')

# Create base output directory
BASE_OUTPUT_DIR="plots/${ALGO,,}_walker_${PERCENTAGE}pct_training_surfaces"

echo "=========================================="
echo "Training Reward Surface Generation Pipeline"
echo "=========================================="
echo "Training run:  ${TRAINING_RUN_DIR}"
echo "Algorithm:     ${ALGO}"
echo "MPC %:         ${PERCENTAGE}%"
echo "Base output:   ${BASE_OUTPUT_DIR}"
echo "Grid size:     ${GRID_SIZE}"
echo "Magnitude:     ${MAGNITUDE}"
echo "Num episodes:  ${NUM_EPISODES}"
echo "CPUs:          ${NUM_CPUS}"
echo "=========================================="
echo ""

# Check if checkpoints directory exists
CHECKPOINTS_DIR="${TRAINING_RUN_DIR}/checkpoints"
if [ ! -d "${CHECKPOINTS_DIR}" ]; then
    echo "Error: Checkpoints directory not found: ${CHECKPOINTS_DIR}"
    exit 1
fi

# Get list of checkpoint model files (sorted by step number)
CHECKPOINT_FILES=($(ls "${CHECKPOINTS_DIR}"/model_*_steps.zip 2>/dev/null | sort -V))

if [ ${#CHECKPOINT_FILES[@]} -eq 0 ]; then
    echo "Error: No checkpoint files found in ${CHECKPOINTS_DIR}"
    exit 1
fi

echo "Found ${#CHECKPOINT_FILES[@]} checkpoints"
echo ""

# =============================================================================
# Process each checkpoint
# =============================================================================

CHECKPOINT_NUM=0
TOTAL_CHECKPOINTS=${#CHECKPOINT_FILES[@]}

for CHECKPOINT_FILE in "${CHECKPOINT_FILES[@]}"; do
    ((CHECKPOINT_NUM++))
    
    # Extract step number from filename (e.g., "model_100000_steps.zip" -> "100000")
    STEPS=$(echo "${CHECKPOINT_FILE}" | grep -oP 'model_\K\d+(?=_steps\.zip)')
    
    echo "=========================================="
    echo "Processing checkpoint ${CHECKPOINT_NUM}/${TOTAL_CHECKPOINTS}"
    echo "=========================================="
    echo "Checkpoint: ${CHECKPOINT_FILE}"
    echo "Steps:      ${STEPS}"
    
    # Create output directory for this checkpoint
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/checkpoint_${STEPS}"
    
    echo "Output:     ${OUTPUT_DIR}"
    
    # Look for corresponding vecnormalize file
    VECNORMALIZE_FILE="${CHECKPOINTS_DIR}/model_vecnormalize_${STEPS}_steps.pkl"
    VECNORMALIZE_ARG=""
    if [ -f "${VECNORMALIZE_FILE}" ]; then
        echo "VecNorm:    ${VECNORMALIZE_FILE}"
        VECNORMALIZE_ARG="--vecnormalize=${VECNORMALIZE_FILE}"
    fi
    echo ""
    
    # Environment name for plot titles
    ENV_NAME="${ALGO} Walker ${PERCENTAGE}pct - ${STEPS} steps"
    
    # Step 1: Generate evaluation jobs
    echo "  [1/4] Generating evaluation jobs..."
    python reward_surfaces/scripts/generate_plane_jobs.py \
        "${CHECKPOINT_FILE}" \
        "${OUTPUT_DIR}" \
        --grid-size="${GRID_SIZE}" \
        --magnitude="${MAGNITUDE}" \
        --num-episodes="${NUM_EPISODES}" \
        ${VECNORMALIZE_ARG}
    
    if [ $? -ne 0 ]; then
        echo "  Error: Job generation failed for checkpoint ${STEPS}"
        continue
    fi
    
    # Step 2: Run evaluations in parallel
    echo "  [2/4] Running ${GRID_SIZE}x${GRID_SIZE} evaluations..."
    python reward_surfaces/scripts/run_jobs_multiproc.py \
        --num-cpus="${NUM_CPUS}" \
        "${OUTPUT_DIR}/jobs.sh"
    
    if [ $? -ne 0 ]; then
        echo "  Error: Evaluation jobs failed for checkpoint ${STEPS}"
        continue
    fi
    
    # Step 3: Aggregate results to CSV
    echo "  [3/4] Aggregating results to CSV..."
    python reward_surfaces/scripts/job_results_to_csv.py \
        "${OUTPUT_DIR}"
    
    if [ $? -ne 0 ]; then
        echo "  Error: Result aggregation failed for checkpoint ${STEPS}"
        continue
    fi
    
    # Step 4: Generate plots
    echo "  [4/4] Generating surface plots..."
    python reward_surfaces/scripts/plot_plane.py \
        "${OUTPUT_DIR}/results.csv" \
        --outname="${OUTPUT_DIR}/surface" \
        --env-name="${ENV_NAME}" \
        --type=all
    
    if [ $? -ne 0 ]; then
        echo "  Error: Plotting failed for checkpoint ${STEPS}"
        continue
    fi
    
    echo "  Checkpoint ${STEPS} complete!"
    echo ""
done

echo ""
echo "=========================================="
echo "All checkpoints processed!"
echo "=========================================="
echo "Results saved to: ${BASE_OUTPUT_DIR}"
echo ""
echo "Checkpoint subdirectories:"
for CHECKPOINT_FILE in "${CHECKPOINT_FILES[@]}"; do
    STEPS=$(echo "${CHECKPOINT_FILE}" | grep -oP 'model_\K\d+(?=_steps\.zip)')
    echo "  - checkpoint_${STEPS}/"
done
echo "=========================================="
