import sbx
import shimmy
import stable_baselines3 as sb3

from dm_control import suite
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.env_checker import check_env

def make_dm_env(render_mode=None):
    """Create a dm_control cartpole environment wrapped for gymnasium."""
    env = suite.load(domain_name="cartpole", task_name="swingup")
    gym_env = FlattenObservation(shimmy.DmControlCompatibilityV0(env, render_mode=render_mode))
    return gym_env

gym_env = sb3.common.env_util.make_vec_env(make_dm_env, n_envs=5)
gym_env = sb3.common.vec_env.VecNormalize(gym_env, norm_obs=True, norm_reward=True)


model = sb3.SAC("MlpPolicy", gym_env, verbose=1).learn(100_000, progress_bar=True)

# Save the model and normalization parameters
model.save("sac_dm_cartpole")
gym_env.save("vec_normalize.pkl")

# Create a new single environment for evaluation
print("\nEvaluating trained policy...")
eval_env_base = make_dm_env(render_mode="human")
eval_env = sb3.common.vec_env.VecNormalize.load("vec_normalize.pkl", 
                                                sb3.common.vec_env.DummyVecEnv([lambda: eval_env_base]))
eval_env.training = False  # Don't update normalization during evaluation
eval_env.norm_reward = False  # Don't normalize rewards during evaluation

# Load the trained model
model = sb3.SAC.load("sac_dm_cartpole")

# Run evaluation episodes
n_eval_episodes = 1
episode_rewards = []

for episode in range(n_eval_episodes):
    obs = eval_env.reset()
    episode_reward = 0
    done = False
    step_count = 0
    
    while not done and step_count < 1000:  # Add step limit
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        episode_reward += reward[0]  # Extract scalar from array
        step_count += 1
        
        if done:
            episode_rewards.append(episode_reward)
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {step_count}")
            break

print(f"\nEvaluation Results:")
print(f"Mean reward: {sum(episode_rewards)/len(episode_rewards):.2f}")
print(f"Min reward: {min(episode_rewards):.2f}")
print(f"Max reward: {max(episode_rewards):.2f}")

eval_env.close()