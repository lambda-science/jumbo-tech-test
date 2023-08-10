import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


seed = 42
max_steps = 50
device = "auto"
log_folder = "./log/"
learning_rate = 1e-4
learning_starts = 0
max_episode = 80000
learning_timesteps = 250 * max_episode
batch_size = 2048
exploration_fraction = 0.8
exploration_initial_eps = 1
exploration_final_eps = 0.05
n_eval_episodes = 100

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
model = DQN("MlpPolicy", env, verbose=1, device=device).load(
    "models/dqn_model",
    env=env,
)

# Evaluate the models without PyGame render, only the reward
mean_reward, _ = evaluate_policy(
    model, env, n_eval_episodes=n_eval_episodes, deterministic=False
)
print("Mean reward: ", mean_reward)

# Evaluate the model visually with PyGame render
env.set_render_mode("human")
mean_reward, _ = evaluate_policy(
    model, env, n_eval_episodes=n_eval_episodes, deterministic=False
)

# Evaluate the model visually with PyGame render
env.set_determinist_mode(False)
mean_reward, _ = evaluate_policy(
    model, env, n_eval_episodes=n_eval_episodes, deterministic=False
)