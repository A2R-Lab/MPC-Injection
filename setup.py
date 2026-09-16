from setuptools import setup, find_packages

setup(
    name="mpc_rl",
    version="0.1.0",
    packages=find_packages(),
    package_data={
        "mpc_rl.planner": ["contracts/mpx_transition_parity/*.json"],
    },
    install_requires=[
        "gymnasium",
        "numpy",
        "jax",
        "sbx-rl",
        "stable-baselines3",
        "dm-control",
        "shimmy",
    ],
    python_requires=">=3.11",
    description="MPC-Injection: Model Predictive Control with Reinforcement Learning",
)
