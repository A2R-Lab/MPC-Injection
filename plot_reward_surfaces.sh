#!/bin/bash

# Generate and plot reward surfaces for trained models
# Complete pipeline: generate jobs -> run evaluations -> aggregate -> plot

# =============================================================================
# Configuration - Modify these variables
# =============================================================================

MODEL_PATH="logs/SAC-MPC-walker-velocity_only_reward/1st_run/walker-walk-SAC-MPC-20251205-102823-percentage-0pct/best_model/"

OUTPUT_DIR="plots/sac_mpc_walker_velocity_only_reward_0_reward_surface/"

ENV_NAME="SAC-MPC Walker Walk 0pct"

# Surface generation parameters
GRID_SIZE=31
MAGNITUDE=0.5
NUM_EPISODES=25
NUM_CPUS=16

# =============================================================================
# Pipeline Execution
# =============================================================================

echo "=========================================="
echo "Reward Surface Generation Pipeline"
echo "=========================================="
echo "Model:        ${MODEL_PATH}"
echo "Output:       ${OUTPUT_DIR}"
echo "Env name:     ${ENV_NAME}"
echo "Grid size:    ${GRID_SIZE}"
echo "Magnitude:    ${MAGNITUDE}"
echo "Num episodes: ${NUM_EPISODES}"
echo "CPUs:         ${NUM_CPUS}"
echo "=========================================="
echo ""

# Step 1: Generate evaluation jobs
echo "Step 1/4: Generating evaluation jobs..."
python reward_surfaces/scripts/generate_plane_jobs.py \
    "${MODEL_PATH}" \
    "${OUTPUT_DIR}" \
    --grid-size="${GRID_SIZE}" \
    --magnitude="${MAGNITUDE}" \
    --num-episodes="${NUM_EPISODES}"

if [ $? -ne 0 ]; then
    echo "Error: Job generation failed"
    exit 1
fi
echo ""

# Step 2: Run evaluations in parallel
echo "Step 2/4: Running ${GRID_SIZE}x${GRID_SIZE} evaluations in parallel..."
python reward_surfaces/scripts/run_jobs_multiproc.py \
    --num-cpus="${NUM_CPUS}" \
    "${OUTPUT_DIR}/jobs.sh"

if [ $? -ne 0 ]; then
    echo "Error: Evaluation jobs failed"
    exit 1
fi
echo ""

# Step 3: Aggregate results to CSV
echo "Step 3/4: Aggregating results to CSV..."
python reward_surfaces/scripts/job_results_to_csv.py \
    "${OUTPUT_DIR}"

if [ $? -ne 0 ]; then
    echo "Error: Result aggregation failed"
    exit 1
fi
echo ""

# Step 4: Generate plots
echo "Step 4/4: Generating surface plots..."
python reward_surfaces/scripts/plot_plane.py \
    "${OUTPUT_DIR}/results.csv" \
    --outname="${OUTPUT_DIR}/surface" \
    --env-name="${ENV_NAME}" \
    --type=all

if [ $? -ne 0 ]; then
    echo "Error: Plotting failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "Reward surface generation complete!"
echo "=========================================="
echo "Results saved to: ${OUTPUT_DIR}"
echo "Plots:"
echo "  - ${OUTPUT_DIR}/surface_3dsurface.png"
echo "  - ${OUTPUT_DIR}/surface_heatmap.png"
echo "  - ${OUTPUT_DIR}/surface_contour.png"
echo "  - ${OUTPUT_DIR}/surface_contourf.png"
echo "=========================================="
