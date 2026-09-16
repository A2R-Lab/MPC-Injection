# MPC-Injection: Biasing Off-Policy Locomotion RL Toward Controller-Induced Behavior Basins

**Roy Xing, Seyoung Ree, and Brian Plancher — CoRL 2026**

MPC-Injection adds controller-generated transitions to off-policy RL replay buffers. The injected experience biases exploration toward controller-induced behavior while the policy learns from the task reward. The repository includes SAC-MPC and TD3-MPC, tagged replay buffers, trajectory generators, and evaluation tools.

| Task family | Entry point |
|---|---|
| 2D walker | `walker-walk` |
| Go2 trotting | `quadruped-velocity_tracking` |
| Go2 barrel roll | `quadruped-barrel_roll` |
| MPX bounding | Bound generator + `quadruped-velocity_tracking` with its recorded action interface |
| Three-legged half-cheetah | `cheetah3-run` |

## Installation

Validated on Linux x86-64 with Python 3.11.13 and Conda. The environment pins JAX 0.6.2, MuJoCo 3.3.6, SBX 0.23.0, and SB3 2.7.0. It includes CUDA 12 JAX packages; CPU smoke checks use `JAX_PLATFORMS=cpu`. Headless rendering uses `MUJOCO_GL=egl` and requires a working EGL driver. Video recording requires `ffmpeg` on `PATH`; interactive viewers also require GLFW/X11 and a display.

Install native prerequisites first: CMake, a C/C++ compiler, OpenGL/X11 development libraries, and zlib development headers. MuJoCo MPC's CMake build downloads its pinned native dependencies. Real-Go2 compilation has additional prerequisites in [REPRODUCING.md](REPRODUCING.md#policy-export-and-real-go2).

From a new parent directory, clone the repository and pinned dependencies.

```bash
git clone --recurse-submodules https://github.com/A2R-Lab/MPC-Injection.git
cd MPC-Injection
conda env create --prefix "$PWD/.conda/mpc-injection" --file environment.yml
conda activate "$PWD/.conda/mpc-injection"
python -m pip install -e deps/gym-quadruped
python -m pip install -e deps/mpx
python -m pip install -e .
cmake -S deps/mujoco_mpc -B deps/mujoco_mpc/build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DMJPC_BUILD_GRPC_SERVICE=ON
cmake --build deps/mujoco_mpc/build --target agent_server ui_agent_server --parallel 2
(cd deps/mujoco_mpc/python && python -m pip install .)
python -m pip check
```

These commands create the Conda prefix, editable package metadata, and `deps/mujoco_mpc/build/`. The MJPC Python installer runs its own CMake build hook; building the server targets first bounds the expensive compilation. Imports retain the name `mpc_rl` for compatibility. No MPX branch switching is needed: trotting/barrel roll retain their original implementation; bounding uses `config_bound` and the bound gait selection.

For a minimal simulation from the repository root (console output only):

```bash
JAX_PLATFORMS=cpu MUJOCO_GL=egl python - <<'PY'
import numpy as np
from mpc_rl.envs.dm_control_env import load_dm_control_env
env = load_dm_control_env("walker", "walk", task_kwargs={"random": 1})
env.reset()
for _ in range(5):
    print(env.step(np.zeros(env.action_spec().shape)).reward)
env.close()
PY
```

See [REPRODUCING.md](REPRODUCING.md) for regeneration, training, evaluation, and deployment. Datasets and pretrained policies are not bundled. This is a completed research artifact with limited maintenance; local startup and integration checks do not establish paper-score reproduction or hardware safety.

## Acknowledgments

The implementation builds on MuJoCo MPC, MPX, gym-quadruped, DM Control, Stable-Baselines3, and SBX. Go2 deployment is based on [Unitree MjLab deployment code](https://github.com/unitreerobotics/unitree_rl_mjlab). The root MIT license and existing dependency/third-party notices are retained.
