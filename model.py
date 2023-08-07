import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# Register our custom environment
gym.register(
    id="JumboEnv-v0",
    entry_point="jumbo_gym:JumboEnv",
    max_episode_steps=200,
)

np.random.seed(42)
# Create the environment
env = gym.make("JumboEnv-v0")
env = Monitor(env)

# Train DQN Model
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=30000)
model.save("dqn_matrix_env")

# Evaluate the models without PyGame render, only the reward
mean_reward_epsilon_greedy, _ = evaluate_policy(model, env, n_eval_episodes=10)
print("Mean reward: ", mean_reward_epsilon_greedy)


# Evaluate the model visually with PyGame render
env.set_render_mode("human")
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
