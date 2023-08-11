import minigrid
import gymnasium as gym
import numpy as np
import torch
from torch import nn
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback


class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    """Custom features extractor for Minigrid to work with Stable-Baselines3.
    From official documentation."""

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
        normalized_image: bool = False,
    ) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


seed = 42
max_steps = 1000
device = "auto"
log_folder = "./log/"
learning_timesteps = 10_000_000
policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)

# Register our custom environment
np.random.seed(seed)
gym.register(
    id="SimpleEnv-v0",
    entry_point="minigrid_env:SimpleEnv",
    max_episode_steps=max_steps,
)

env = gym.make(
    "SimpleEnv-v0", render_mode="rgb_array", determinist=False, max_steps=max_steps
)
env = ImgObsWrapper(env)

checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path="./models/",
    name_prefix="ppo_minigrid_model_random_map",
    save_replay_buffer=False,
    save_vecnormalize=True,
)

# Create PPO Model with parameters
model = PPO(
    "CnnPolicy",
    env,
    tensorboard_log=log_folder,
    device=device,
    policy_kwargs=policy_kwargs,
    verbose=1,
)

# Reload previous model as starting point
model.set_parameters("models/ppo_minigrid_model_determinist_map")

# Learn the model
model.learn(
    total_timesteps=learning_timesteps,
    callback=checkpoint_callback,
    progress_bar=True,
)
model.save("models/ppo_minigrid_model_random_map")
