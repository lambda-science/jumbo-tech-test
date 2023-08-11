import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

seed = 42
max_steps = 1000
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

checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path="./models/",
    name_prefix="dqn_model_determinist_map",
    save_replay_buffer=False,
    save_vecnormalize=True,
)

# Create DQN Model with parameters
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

# Train the model
model.learn(
    total_timesteps=learning_timesteps,
    callback=checkpoint_callback,
    progress_bar=True,
)
model.save("models/dqn_model_determinist_map")
model.save_replay_buffer("models/dqn_model_determinist_map_replay_buffer")
