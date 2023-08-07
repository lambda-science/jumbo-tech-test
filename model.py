import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

MAX_STEPS = 200
SEED = 42
N_EVAL_EPISODES = 10
LEARNING_TIMESTEPS = 30000

# Register our custom environment
gym.register(
    id="JumboEnv-v0",
    entry_point="jumbo_gym:JumboEnv",
    max_episode_steps=MAX_STEPS,
)

np.random.seed(SEED)
# Create the environment
env = gym.make("JumboEnv-v0")
env = Monitor(env)

# Train DQN Model
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=LEARNING_TIMESTEPS)
model.save("dqn_matrix_env")

# Evaluate the models without PyGame render, only the reward
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=N_EVAL_EPISODES)
print("Mean reward: ", mean_reward)


# Evaluate the model visually with PyGame render
env.set_render_mode("human")
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=N_EVAL_EPISODES)
