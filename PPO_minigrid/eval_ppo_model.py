import minigrid
import gymnasium as gym
import numpy as np
import torch
from torch import nn
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.evaluation import evaluate_policy


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


def evaluate_model(model, env, n_eval_episodes):
    """Evaluate the model on the given number of episodes and
    return the mean reward and length"""
    reward, ep_length = evaluate_policy(
        model, env, n_eval_episodes=n_eval_episodes, deterministic=False
    )
    return np.mean(reward), np.mean(ep_length)


seed = 42
max_steps = 50
device = "auto"
log_folder = "./log/"
n_eval_episodes = 10
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

env = gym.make("SimpleEnv-v0", render_mode="human", determinist=True)
env = ImgObsWrapper(env)

# Model Loading
model = PPO.load(
    "models/ppo_minigrid_model_random_map",
    env=env,
)

# Evaluate the models on deterministic map
mean_reward, mean_length = evaluate_model(model, env, n_eval_episodes)
print(f"Mean reward (determinist): {mean_reward} Mean Length: {mean_length}")

# Evaluate the model on random map
env.set_determinist_mode(False)
mean_reward, mean_length = evaluate_model(model, env, n_eval_episodes)
print(f"Mean reward (determinist): {mean_reward} Mean Length: {mean_length}")
