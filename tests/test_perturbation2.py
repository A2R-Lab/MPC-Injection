#!/usr/bin/env python3
"""
Test to see the actual effect of the perturbation over multiple steps
"""

import numpy as np
from mpc_rl.envs.dm_control_env import load_dm_control_env
from shimmy import DmControlCompatibilityV0
from gymnasium.wrappers import FlattenObservation

# Create environment
dm_env = load_dm_control_env(domain_name='walker', task_name='walk')
gym_env = DmControlCompatibilityV0(dm_env, render_mode=None)
gym_env = FlattenObservation(gym_env)

# Reset and get physics
obs, _ = gym_env.reset(seed=42)
physics = gym_env.unwrapped._env.physics

# Get torso body ID
torso_body_id = physics.model.name2id('torso', 'body')

print("Testing perturbation effect over time:")
print("="*80)

# Scenario 1: No perturbation (baseline)
print("\nScenario 1: No perturbation")
obs, _ = gym_env.reset(seed=42)
positions_no_pert = []
for i in range(20):
    positions_no_pert.append(physics.named.data.xpos['torso'].copy())
    action = np.zeros(gym_env.action_space.shape)
    obs, reward, done, truncated, info = gym_env.step(action)
positions_no_pert = np.array(positions_no_pert)
print(f"Position at step 0: {positions_no_pert[0]}")
print(f"Position at step 10: {positions_no_pert[10]}")
print(f"Position at step 19: {positions_no_pert[19]}")
print(f"Total X displacement: {positions_no_pert[19][0] - positions_no_pert[0][0]:.6f}")

# Scenario 2: Perturbation applied once at step 10
print("\n" + "="*80)
print("Scenario 2: Perturbation of -100N applied ONCE at step 10")
obs, _ = gym_env.reset(seed=42)
positions_pert_once = []
for i in range(20):
    positions_pert_once.append(physics.named.data.xpos['torso'].copy())
    
    # Apply perturbation at step 10, then clear it at step 11
    if i == 10:
        physics.data.xfrc_applied[torso_body_id] = [-100.0, 0, 0, 0, 0, 0]
        print(f"  -> Applied perturbation at step {i}")
    elif i == 11:
        physics.data.xfrc_applied[torso_body_id] = [0, 0, 0, 0, 0, 0]
        print(f"  -> Cleared perturbation at step {i}")
    
    action = np.zeros(gym_env.action_space.shape)
    obs, reward, done, truncated, info = gym_env.step(action)
    
positions_pert_once = np.array(positions_pert_once)
print(f"Position at step 0: {positions_pert_once[0]}")
print(f"Position at step 10: {positions_pert_once[10]}")
print(f"Position at step 11: {positions_pert_once[11]}")
print(f"Position at step 19: {positions_pert_once[19]}")
print(f"Total X displacement: {positions_pert_once[19][0] - positions_pert_once[0][0]:.6f}")
print(f"Difference from baseline: {(positions_pert_once[19][0] - positions_pert_once[0][0]) - (positions_no_pert[19][0] - positions_no_pert[0][0]):.6f}")

# Scenario 3: Perturbation applied and kept for 10 steps
print("\n" + "="*80)
print("Scenario 3: Perturbation of -100N applied and HELD for 10 steps (10-19)")
obs, _ = gym_env.reset(seed=42)
positions_pert_held = []
for i in range(20):
    positions_pert_held.append(physics.named.data.xpos['torso'].copy())
    
    # Apply perturbation at step 10 and keep it until step 20
    if i == 10:
        physics.data.xfrc_applied[torso_body_id] = [-100.0, 0, 0, 0, 0, 0]
        print(f"  -> Applied perturbation at step {i}")
    
    action = np.zeros(gym_env.action_space.shape)
    obs, reward, done, truncated, info = gym_env.step(action)
    
positions_pert_held = np.array(positions_pert_held)
print(f"Position at step 0: {positions_pert_held[0]}")
print(f"Position at step 10: {positions_pert_held[10]}")
print(f"Position at step 19: {positions_pert_held[19]}")
print(f"Total X displacement: {positions_pert_held[19][0] - positions_pert_held[0][0]:.6f}")
print(f"Difference from baseline: {(positions_pert_held[19][0] - positions_pert_held[0][0]) - (positions_no_pert[19][0] - positions_no_pert[0][0]):.6f}")

print("\n" + "="*80)
print("Summary:")
print(f"No perturbation: X change = {positions_no_pert[19][0] - positions_no_pert[0][0]:.6f}")
print(f"Single-step perturbation: X change = {positions_pert_once[19][0] - positions_pert_once[0][0]:.6f}")
print(f"10-step held perturbation: X change = {positions_pert_held[19][0] - positions_pert_held[0][0]:.6f}")
