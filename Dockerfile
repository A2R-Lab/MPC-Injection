# CUDA 12 runtime base for JAX CUDA, MuJoCo, and NVIDIA Slurm/AppContainer use.
# Use Ubuntu 24.04 to match your project notes.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu24.04

SHELL ["/bin/bash", "-lc"]

ENV DEBIAN_FRONTEND=noninteractive

# ---- System dependencies ----
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

# ---- Conda via Miniforge ----
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}

RUN wget -qO /tmp/miniforge.sh \
      https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && bash /tmp/miniforge.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniforge.sh \
    && conda config --system --set channel_priority strict \
    && conda clean -afy

# ---- MuJoCo / JAX / headless settings ----
# MUJOCO_GL=egl is the default for headless GPU rendering.
# PYOPENGL_PLATFORM=egl avoids accidental GLX/X11 paths.
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV XLA_PYTHON_CLIENT_ALLOCATOR=platform
ENV JAX_PLATFORMS=cuda
ENV XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda

# Optional but useful for reproducible logs.
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /workspace/MPC-RL

# Copy only the environment file first so Docker can cache the conda env layer.
COPY environment.yml /workspace/MPC-RL/environment.yml

# Create exactly the conda env requested by your project.
RUN conda env create -f environment.yml \
    && conda clean -afy

# Make subsequent RUN commands use the conda env.
ENV CONDA_DEFAULT_ENV=mpc-rl
ENV PATH=${CONDA_DIR}/envs/mpc-rl/bin:${CONDA_DIR}/bin:${PATH}

# Copy the full repo after the conda env is built.
COPY . /workspace/MPC-RL

# Install local project packages.
# We intentionally do NOT build deps/mujoco_mpc.
RUN python -m pip install --upgrade pip \
    && python -m pip install -e . \
    && python -m pip install -e deps/gym-quadruped \
    && python -m pip install -e deps/mpx

# Basic import smoke test at build time.
RUN python - <<'PY'
import sys
print("Python:", sys.version)
import jax
print("JAX backend:", jax.default_backend())
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

# Default command is an interactive shell in the conda env.
CMD ["/bin/bash"]
