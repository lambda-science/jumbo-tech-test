import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

SEED = 42
MAX_STEPS = 200
DEVICE = "auto"
LOG_FOLDER = "./log/"
LEARNING_RATE = 1e-4
LEARNING_TIMESTEPS = 100_000
LEARNING_STARTS = 50_000
N_EVAL_EPISODES = 10
BATCH_SIZE = 2048
EXPLORAITON_FRACTION = 0.1
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
env = gym.make("JumboEnv-v0")
env = Monitor(env)

# Train DQN Model
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=LEARNING_RATE,
    learning_starts=LEARNING_STARTS,
    batch_size=BATCH_SIZE,
    exploration_fraction=EXPLORAITON_FRACTION,
    exploration_final_eps=EXPLORATION_FINAL_EPS,
    verbose=1,
    seed=SEED,
    device=DEVICE,
    tensorboard_log=LOG_FOLDER,
)
model.learn(total_timesteps=LEARNING_TIMESTEPS)
model.save("dqn_matrix_env")

# Evaluate the models without PyGame render, only the reward
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=N_EVAL_EPISODES)
print("Mean reward: ", mean_reward)


# Evaluate the model visually with PyGame render
env.set_render_mode("human")
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=N_EVAL_EPISODES)
