import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


seed = 42
max_steps = 1000
device = "auto"
log_folder = "./log/"
learning_rate = 1e-4
learning_starts = 0
max_episode = 80000
learning_timesteps = 250 * max_episode
batch_size = 256
exploration_fraction = 0.8
exploration_initial_eps = 1
exploration_final_eps = 0.05
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
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=learning_rate,
    learning_starts=learning_starts,
    batch_size=batch_size,
    exploration_fraction=exploration_fraction,
    exploration_initial_eps=exploration_initial_eps,
    exploration_final_eps=exploration_final_eps,
    verbose=1,
    seed=seed,
    device=device,
    tensorboard_log=log_folder,
)
model.load(
    "models/models/dqn_model",
    env=env,
)

# Evaluate the models without PyGame render, only the reward
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)
print("Mean reward: ", mean_reward)

# Evaluate the model visually with PyGame render
env.set_render_mode("human")
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)

# Evaluate the model visually with PyGame render
env.set_determinist_mode(False)
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)
