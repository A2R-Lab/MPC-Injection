#!/usr/bin/env python3
"""
Test script to verify MuJoCo xfrc_applied behavior
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

print("Testing MuJoCo xfrc_applied behavior:")
print("="*60)

# Test 1: Set force and check if it persists
print("\nTest 1: Does xfrc_applied persist across steps?")
print(f"Initial xfrc_applied[{torso_body_id}]: {physics.data.xfrc_applied[torso_body_id]}")

# Apply a large force
test_force = -100.0
physics.data.xfrc_applied[torso_body_id] = [test_force, 0, 0, 0, 0, 0]
print(f"After setting: {physics.data.xfrc_applied[torso_body_id]}")

# Take a step with zero action
action = np.zeros(gym_env.action_space.shape)
obs, reward, done, truncated, info = gym_env.step(action)

# Check if force is still there
print(f"After env.step(): {physics.data.xfrc_applied[torso_body_id]}")
print(f"Result: Force was {'CLEARED' if np.allclose(physics.data.xfrc_applied[torso_body_id], 0) else 'PRESERVED'}")

# Test 2: Record torso position before and after applying force
print("\n" + "="*60)
print("Test 2: Does applying force just before step() have effect?")
print("="*60)

# Reset environment
obs, _ = gym_env.reset(seed=42)
physics = gym_env.unwrapped._env.physics

# Record initial torso position
initial_pos = physics.named.data.xpos['torso'].copy()
print(f"Initial torso position: {initial_pos}")

# Run a few steps without perturbation
for i in range(10):
    action = np.zeros(gym_env.action_space.shape)
    obs, reward, done, truncated, info = gym_env.step(action)

pos_before = physics.named.data.xpos['torso'].copy()
vel_before = physics.named.data.sensordata.copy()  # Get velocity data

print(f"Torso position before perturbation: {pos_before}")

# Apply perturbation JUST BEFORE step
physics.data.xfrc_applied[torso_body_id] = [test_force, 0, 0, 0, 0, 0]
print(f"Applied force: {physics.data.xfrc_applied[torso_body_id]}")

# Step with zero action
action = np.zeros(gym_env.action_space.shape)
obs, reward, done, truncated, info = gym_env.step(action)

pos_after = physics.named.data.xpos['torso'].copy()
print(f"Torso position after step: {pos_after}")
print(f"Position change: {pos_after - pos_before}")
print(f"xfrc_applied after step: {physics.data.xfrc_applied[torso_body_id]}")

print("\n" + "="*60)
print("Conclusion:")
if np.allclose(physics.data.xfrc_applied[torso_body_id], 0):
    print("✓ MuJoCo DOES clear xfrc_applied after each step")
    print("  The force must be set immediately before env.step()")
else:
    print("✗ xfrc_applied persists (unexpected)")
