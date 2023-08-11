import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

seed = 42
max_steps = 50
device = "auto"
n_eval_episodes = 10

# Register our custom environment
gym.register(
    id="JumboEnv-v0",
    entry_point="jumbo_gym:JumboEnv",
    max_episode_steps=max_steps,
)

np.random.seed(seed)
# Create the environment
env = gym.make("JumboEnv-v0", determinist=True)
env = Monitor(env)

# Post-Exploitation Model Loading
model = DQN.load(
    "models/dqn_model_determinist_map",
    env=env,
)

# Evaluate the models without render, only the reward, on deterministic map
mean_reward, _ = evaluate_policy(
    model, env, n_eval_episodes=n_eval_episodes, deterministic=False
)
print("Mean reward (determinist): ", mean_reward)

# Evaluate the models WITH render, on deterministic map
env.set_render_mode("human")
mean_reward, _ = evaluate_policy(
    model, env, n_eval_episodes=n_eval_episodes, deterministic=False
)

# Evaluate the models without render, only the reward, on random map
env.set_render_mode("rgb_array")
env.set_determinist_mode(False)
mean_reward, _ = evaluate_policy(
    model, env, n_eval_episodes=n_eval_episodes, deterministic=False
)
print("Mean reward (random map): ", mean_reward)

# Evaluate the models WITH render, on random map
env.set_render_mode("human")
mean_reward, _ = evaluate_policy(
    model, env, n_eval_episodes=n_eval_episodes, deterministic=False
)
