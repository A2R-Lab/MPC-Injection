# CUDA 13 + Ubuntu 24.04 base.
# This matches your stated preference: Ubuntu 24 + CUDA 13.
FROM nvidia/cuda:13.2.1-cudnn-runtime-ubuntu24.04

SHELL ["/bin/bash", "-lc"]

ENV DEBIAN_FRONTEND=noninteractive

# -----------------------------------------------------------------------------
# System dependencies
# -----------------------------------------------------------------------------
# ffmpeg: MP4/video writing via mediapy.
# libGL / EGL / GLFW / OSMesa / X11 libs: MuJoCo, OpenCV, dm_control rendering.
# build-essential/pkg-config/patchelf: native extension + robotics/sim packages.
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    pkg-config \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    libglfw3 \
    libglew2.2 \
    libosmesa6 \
    libegl1 \
    libx11-6 \
    libxrandr2 \
    libxi6 \
    libxcursor1 \
    libxinerama1 \
    libxxf86vm1 \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Miniforge / conda
# -----------------------------------------------------------------------------
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}

RUN wget -qO /tmp/miniforge.sh \
      https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && bash /tmp/miniforge.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniforge.sh \
    && conda config --system --set channel_priority strict \
    && conda clean -afy

# -----------------------------------------------------------------------------
# Runtime defaults
# -----------------------------------------------------------------------------
# Use EGL for headless MuJoCo rendering.
# Keep JAX GPU settings for actual docker run / apptainer exec runtime.
# Do not rely on these during docker build, because docker build normally has
# no GPU device exposed.
# -----------------------------------------------------------------------------
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV XLA_PYTHON_CLIENT_ALLOCATOR=platform
ENV JAX_PLATFORMS=cuda
ENV XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /workspace/MPC-RL

# -----------------------------------------------------------------------------
# Copy environment first for Docker layer caching.
# -----------------------------------------------------------------------------
COPY environment.yml /workspace/MPC-RL/environment.yml

# -----------------------------------------------------------------------------
# Patch the environment file for CUDA 13.
#
# Your source environment.yml currently uses:
#   - jax[cuda12]==0.6.2
#   - jaxlib==0.6.2
#
# That is inconsistent with a CUDA 13 base image. For this Docker image only,
# generate environment.docker.yml where:
#   - jax[cuda12]==0.6.2 is replaced by jax[cuda13]
#   - jaxlib==0.6.2 is removed, because the jax[cuda13] extra should resolve
#     the compatible jaxlib plugin stack.
#
# This leaves your source environment.yml unchanged on disk.
# -----------------------------------------------------------------------------
RUN python - <<'PY'
from pathlib import Path

src = Path("environment.yml")
dst = Path("environment.docker.yml")

text = src.read_text()

text = text.replace("      - jax[cuda12]==0.6.2\n", "      - jax[cuda13]\n")
text = text.replace("      - jaxlib==0.6.2\n", "")

dst.write_text(text)
print(dst.read_text())
PY

# Create the conda environment.
RUN conda env create -f environment.docker.yml \
    && conda clean -afy

# Make all subsequent commands use the conda environment.
ENV CONDA_DEFAULT_ENV=mpc-rl
ENV PATH=${CONDA_DIR}/envs/mpc-rl/bin:${CONDA_DIR}/bin:${PATH}

# -----------------------------------------------------------------------------
# Copy the full repo.
# Make sure .dockerignore excludes data/, logs/, jax_cache/, videos/, etc.
# -----------------------------------------------------------------------------
COPY . /workspace/MPC-RL

# -----------------------------------------------------------------------------
# Install local editable packages.
# We intentionally do NOT build deps/mujoco_mpc, per your instruction.
# -----------------------------------------------------------------------------
RUN python -m pip install --upgrade pip \
    && python -m pip install -e . \
    && python -m pip install -e deps/gym-quadruped \
    && python -m pip install -e deps/mpx

# -----------------------------------------------------------------------------
# Build-time smoke test.
#
# Force CPU here. docker build does not expose GPU devices, so calling
# jax.default_backend() with JAX_PLATFORMS=cuda during build can fail even if the
# final runtime image is correct.
# -----------------------------------------------------------------------------
RUN JAX_PLATFORMS=cpu python - <<'PY'
import sys
print("Python:", sys.version)

import jax
print("JAX import OK")

import mujoco
print("MuJoCo:", mujoco.__version__)

import gymnasium
import stable_baselines3
import sbx
import mediapy
import cv2
import mpc_rl
import gym_quadruped
import mpx

print("Core imports OK")
PY

CMD ["/bin/bash"]
