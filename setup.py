from setuptools import setup, find_packages

setup(
    name="mpc_rl",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gymnasium",
        "numpy",
        "jax",
        "sbx-rl",
        "stable-baselines3",
        "dm-control",
        "shimmy",
    ],
    python_requires=">=3.8",
    description="MPC-Injection: Model Predictive Control with Reinforcement Learning",
)
