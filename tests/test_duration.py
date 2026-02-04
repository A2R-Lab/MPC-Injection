#!/usr/bin/env python3
"""
Test the updated perturbation logic with duration support
"""

import sys
from pathlib import Path
import numpy as np
from dm_control import suite
from shimmy import DmControlCompatibilityV0
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def make_dm_env(domain: str, task: str, render_mode='rgb_array', seed=None):
    """Create a dm_control environment wrapped for gymnasium."""
    dm_env = suite.load(domain_name=domain, task_name=task)
    gym_env = DmControlCompatibilityV0(dm_env, render_mode=render_mode)
    gym_env = FlattenObservation(gym_env)
    if seed is not None:
        gym_env.reset(seed=seed)
    return gym_env

# Create simple environment
def env_fn():
    return make_dm_env('walker', 'walk', render_mode=None)

vec_env = DummyVecEnv([env_fn])
vec_env.seed(42)
obs = vec_env.reset()

# Access physics
base_env = vec_env.envs[0].unwrapped
physics = base_env._env.physics
torso_body_id = physics.model.name2id('torso', 'body')

print("Testing perturbation with duration support")
print("="*80)

# Test parameters
PERTURBATION_FORCE = -100.0
PERTURBATION_START = 10
PERTURBATION_DURATION = 5
max_steps = 30

positions = []
forces_applied = []

for step in range(max_steps):
    # Record position
    positions.append(physics.named.data.xpos['torso'].copy())
    
    # Apply perturbation logic (mimicking the updated code)
    perturbation_end_step = PERTURBATION_START + PERTURBATION_DURATION
    
    if PERTURBATION_START <= step < perturbation_end_step:
        physics.data.xfrc_applied[torso_body_id] = [PERTURBATION_FORCE, 0, 0, 0, 0, 0]
        forces_applied.append(PERTURBATION_FORCE)
        if step == PERTURBATION_START:
            print(f"Step {step}: Started applying {PERTURBATION_FORCE}N force")
    elif step >= perturbation_end_step:
        physics.data.xfrc_applied[torso_body_id] = [0, 0, 0, 0, 0, 0]
        forces_applied.append(0.0)
        if step == perturbation_end_step:
            print(f"Step {step}: Cleared force")
    else:
        forces_applied.append(0.0)
    
    # Step with zero action
    action = np.zeros(vec_env.action_space.shape)
    obs, reward, done, info = vec_env.step(action)

positions = np.array(positions)
forces_applied = np.array(forces_applied)

print(f"\nPositions and velocities:")
print("-"*80)
print(f"{'Step':>5} {'X Position':>12} {'X Velocity':>12} {'Force Applied':>15}")
print("-"*80)

for i in range(len(positions)):
    if i > 0:
        velocity = positions[i][0] - positions[i-1][0]
    else:
        velocity = 0.0
    
    marker = ""
    if i == PERTURBATION_START:
        marker = " <-- Start"
    elif i == PERTURBATION_START + PERTURBATION_DURATION:
        marker = " <-- End"
    
    print(f"{i:5d} {positions[i][0]:12.6f} {velocity:12.6f} {forces_applied[i]:15.1f}{marker}")

# Calculate effect
pre_pert_vel = np.mean([positions[i+1][0] - positions[i][0] 
                        for i in range(PERTURBATION_START-3, PERTURBATION_START)])
during_pert_vel = np.mean([positions[i+1][0] - positions[i][0] 
                           for i in range(PERTURBATION_START, PERTURBATION_START+PERTURBATION_DURATION-1)])
post_pert_vel = np.mean([positions[i+1][0] - positions[i][0] 
                         for i in range(PERTURBATION_START+PERTURBATION_DURATION, 
                                       min(PERTURBATION_START+PERTURBATION_DURATION+3, len(positions)-1))])

print(f"\nSummary:")
print(f"  Pre-perturbation velocity:    {pre_pert_vel:8.6f} m/step")
print(f"  During perturbation velocity: {during_pert_vel:8.6f} m/step")
print(f"  Post-perturbation velocity:   {post_pert_vel:8.6f} m/step")
print(f"  Velocity reduction: {(pre_pert_vel - during_pert_vel)/pre_pert_vel * 100:.1f}%")

vec_env.close()
