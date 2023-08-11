import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


def evaluate_model(model, env, n_eval_episodes):
    """Evaluate the model on the given number of episodes and
    return the mean reward and length"""
    reward, ep_length = evaluate_policy(
        model, env, n_eval_episodes=n_eval_episodes, deterministic=False
    )
    return np.mean(reward), np.mean(ep_length)


seed = 42
max_steps = 25
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
env = gym.make("JumboEnv-v0", determinist=True, render_mode="human")
env = Monitor(env)

# Model Loading
model = DQN.load(
    "models/dqn_model_determinist_map",
    env=env,
)


# Evaluate the models on deterministic map
mean_reward, mean_length = evaluate_model(model, env, n_eval_episodes)
print(f"Mean reward (determinist): {mean_reward} Mean Length: {mean_length}")

# Evaluate the model on random map
env.set_determinist_mode(False)
mean_reward, mean_length = evaluate_model(model, env, n_eval_episodes)
print(f"Mean reward (determinist): {mean_reward} Mean Length: {mean_length}")
