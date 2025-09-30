import gymnasium as gym
from dm_control import suite
from shimmy import DmControlCompatibilityV0
from sbx import SAC, PPO, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import os

# Create dm_control cartpole environment
def make_dm_cartpole(render_mode=None):
    """Create a dm_control cartpole environment wrapped for gymnasium."""
    dm_env = suite.load(domain_name="cartpole", task_name="swingup")
    gym_env = DmControlCompatibilityV0(dm_env, render_mode=render_mode)
    return gym_env

def save_video(frames, video_path, fps=30):
    """Save frames as MP4 video using matplotlib."""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    
    fig, ax = plt.subplots()
    ax.axis('off')
    
    def animate(frame_idx):
        ax.clear()
        ax.imshow(frames[frame_idx])
        ax.axis('off')
        return []
    
    ani = animation.FuncAnimation(fig, animate, frames=len(frames), 
                                interval=1000/fps, blit=True, repeat=False)
    
    # Save as MP4
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='SB3'), bitrate=1800)
    ani.save(video_path, writer=writer)
    plt.close(fig)

def main():
    # Training with vectorized environments
    print("Creating vectorized training environments...")
    vec_env = make_vec_env(make_dm_cartpole, n_envs=8)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    print("Training PPO policy...")
    model = PPO(
        "MlpPolicy", 
        vec_env, 
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./ppo_dm_cartpole_tensorboard/"
    )

    model.learn(total_timesteps=100000, progress_bar=True)

    # Save the trained model and normalization parameters
    model.save("ppo_dm_cartpole")
    vec_env.save("vec_normalize.pkl")

    # Evaluation in single environment with video recording
    print("\nEvaluating trained policy...")
    eval_env_base = make_dm_cartpole(render_mode="rgb_array")
    eval_env = VecNormalize.load("vec_normalize.pkl", DummyVecEnv([lambda: eval_env_base]))
    eval_env.training = False  # Don't update normalization during evaluation
    eval_env.norm_reward = False  # Don't normalize rewards during evaluation

    # Load the trained model for evaluation
    model = PPO.load("ppo_dm_cartpole")

    # Create video directory
    video_dir = "evaluation_videos"
    os.makedirs(video_dir, exist_ok=True)

    # Run evaluation episodes with video recording
    n_eval_episodes = 3  # Reduced for video recording
    episode_rewards = []

    for episode in range(n_eval_episodes):
        obs = eval_env.reset()
        episode_reward = 0
        done = False
        frames = []
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            episode_reward += reward[0]  # Extract scalar from array
            
            # Capture frame for video
            frame = eval_env_base.render()
            if frame is not None:
                frames.append(frame)
            
            if done:
                episode_rewards.append(episode_reward)
                print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
                
                # Save video for this episode
                if frames:
                    video_path = os.path.join(video_dir, f"episode_{episode + 1}.mp4")
                    save_video(frames, video_path)
                    print(f"Video saved to: {video_path}")
                break

    print(f"\nEvaluation Results:")
    print(f"Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Min reward: {np.min(episode_rewards):.2f}")
    print(f"Max reward: {np.max(episode_rewards):.2f}")

if __name__ == "__main__":
    main()