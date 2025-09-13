# MPC-RL Experimental Workspace
This repo is a test space for research ideas in combining MPC and RL methods to leverage both their strengths for robot control.

The initial setup is to use MuJoCo Playground and Brax to test state-of-the-art RL algorithms. We then aim to experiment with trajectory optimization techniques and explore how to combine them.

## Structure
The [FastTD3](https://github.com/younggyoseo/FastTD3) project has a nice general structure to refer from in terms of how to write RL algorithms for different frameworks and simulators. However, the GATO-RL repo has more readable file structure and abstraction. We will probably do something similar to both.

## Prereqs
Make sure you have Conda (for environment management) installed. Regular Python virtual environments should also work fine, but the direct setup is for Conda environments.

## Installation
Create the environment from the `environment.yml` file:

``` bash
conda env create -f environment.yml
```

Active the environment with:

``` bash
conda activate mpc_rl_exp
```

If you want to install with a regular Python virtual environment you need to install the following (assuming Python=3.10):
    - jax
    - mujoco
    - mujoco_mjx
    - brax
    - mediapy
    - playground