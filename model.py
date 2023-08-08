import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes

SEED = 42
MAX_STEPS = 1000
DEVICE = "auto"
LOG_FOLDER = "./log/"
LEARNING_RATE = 1e-4
MAX_EPISODE = 100_000
LEARNING_TIMESTEPS = MAX_STEPS * MAX_EPISODE
LEARNING_STARTS = 0
N_EVAL_EPISODES = 10
BATCH_SIZE = 2048
EXPLORATION_FRACTION = 0.9
TRAIN_FREQ = 4
EXPLORATION_FINAL_EPS = 0.05


# Register our custom environment
gym.register(
    id="JumboEnv-v0",
    entry_point="jumbo_gym:JumboEnv",
    max_episode_steps=MAX_STEPS,
)

np.random.seed(SEED)
# Create the environment
env = gym.make("JumboEnv-v0", determinist=True)
env = Monitor(env)
callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=MAX_EPISODE, verbose=1)
# Train DQN Model
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=LEARNING_RATE,
    learning_starts=LEARNING_STARTS,
    batch_size=BATCH_SIZE,
    exploration_fraction=EXPLORATION_FRACTION,
    exploration_final_eps=EXPLORATION_FINAL_EPS,
    verbose=1,
    seed=SEED,
    device=DEVICE,
    tensorboard_log=LOG_FOLDER,
)
model.learn(total_timesteps=LEARNING_TIMESTEPS, callback=callback_max_episodes)
model.save("dqn_matrix_env")

# Evaluate the models without PyGame render, only the reward
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=N_EVAL_EPISODES)
print("Mean reward: ", mean_reward)

# Evaluate the model visually with PyGame render
env.set_render_mode("human")
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=100)

# # Evaluate the model visually with PyGame render
# env.set_determinist_mode(False)
# mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=N_EVAL_EPISODES)
