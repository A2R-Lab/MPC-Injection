# Deployment Code for the Go2

This code is based on the Unitree MjLab deployment code.

NOTE: The policy we want to deploy is the following

```
quadruped-velocity_tracking-SAC-20260305-170808
```

which was trained with the following settings:
```json
{
  "algorithm": "SAC",
  "learning_rate": 0.0003,
  "buffer_size": 1000000,
  "learning_starts": 50000,
  "batch_size": 256,
  "tau": 0.005,
  "gamma": 0.99,
  "gradient_steps": -1,
  "seed": 1,
  "tensorboard_log": "logs/quadruped-velocity_tracking-SAC-20260305-170808/tensorboard",
  "inject_n_timesteps": 5000,
  "inject_type": "percentage",
  "percentage": 25,
  "num_traj": 10,
  "random_select": true,
  "data_dir": null,
  "env_name": "quadruped-velocity_tracking",
  "domain": "quadruped",
  "task": "velocity_tracking",
  "total_timesteps": 5000000,
  "num_envs": 512,
  "save_replay_buffer_checkpoints": false,
  "save_replay_buffer_final": false,
  "domain_randomization": {
    "enabled": true,
    "obs_noise_level": 1.0
  }
}
```
and command:
```
python mpc_rl/train.py --env_name quadruped-velocity_tracking --algorithm SAC --total_timesteps 5000000 --num_envs 512 --seed 1 --learning_starts 50000 --save_replay_buffer_checkpoints=False --save_replay_buffer_final=False --domain_rand=True
```

Trying with bigger `batch_size=512` now...