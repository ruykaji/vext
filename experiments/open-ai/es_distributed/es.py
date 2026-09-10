"""Evolution Strategies optimizer from Salimans et al. (2017)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from .policy import MLPPolicy


def centered_ranks(values: np.ndarray) -> np.ndarray:
    """Map values to evenly spaced ranks in [-0.5, 0.5]."""
    flat = values.ravel()
    if flat.size < 2:
        return np.zeros_like(values, dtype=np.float32)
    ranks = np.empty(flat.size, dtype=np.int64)
    ranks[np.argsort(flat)] = np.arange(flat.size)
    return (ranks.reshape(values.shape) / (flat.size - 1) - 0.5).astype(np.float32)


@dataclass
class Adam:
    size: int
    learning_rate: float
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8

    def __post_init__(self):
        self.iteration = 0
        self.first = np.zeros(self.size, dtype=np.float32)
        self.second = np.zeros(self.size, dtype=np.float32)

    def step(self, gradient: np.ndarray) -> np.ndarray:
        self.iteration += 1
        self.first = self.beta1 * self.first + (1 - self.beta1) * gradient
        self.second = self.beta2 * self.second + (1 - self.beta2) * np.square(gradient)
        scale = self.learning_rate * np.sqrt(1 - self.beta2 ** self.iteration) / (1 - self.beta1 ** self.iteration)
        return scale * self.first / (np.sqrt(self.second) + self.epsilon)


class EvolutionStrategy:
    """Mirrored-noise ES with centered-rank fitness shaping and Adam."""

    def __init__(self, env_factory: Callable[..., gym.Env], policy: MLPPolicy, *, population_size: int = 64,
                 noise_std: float = 0.02, learning_rate: float = 0.01,
                 l2_coefficient: float = 0.005, action_noise_std: float = 0.0,
                 observation_update_probability: float = 0.0, seed: int = 0):
        if population_size < 2:
            raise ValueError("population_size must be at least 2")
        if noise_std <= 0:
            raise ValueError("noise_std must be positive")
        self.env_factory = env_factory
        self.policy = policy
        self.population_size = population_size
        self.noise_std = noise_std
        self.l2_coefficient = l2_coefficient
        self.action_noise_std = action_noise_std
        self.observation_update_probability = observation_update_probability
        self.rng = np.random.default_rng(seed)
        self.optimizer = Adam(policy.num_params, learning_rate)
        self.episodes = 0
        self.observation_sum = np.zeros_like(policy.observation_mean, dtype=np.float64)
        self.observation_sum_of_squares = np.full_like(policy.observation_mean, 1e-2, dtype=np.float64)
        self.observation_count = 1e-2

    def _accumulate_observation_statistics(self, observations: np.ndarray) -> None:
        self.observation_sum += observations.sum(axis=0)
        self.observation_sum_of_squares += np.square(observations).sum(axis=0)
        self.observation_count += len(observations)

    def _apply_observation_statistics(self) -> None:
        mean = self.observation_sum / self.observation_count
        variance = self.observation_sum_of_squares / self.observation_count - np.square(mean)
        self.policy.observation_mean[:] = mean
        self.policy.observation_std[:] = np.sqrt(np.maximum(variance, 1e-2))

    def train_iteration(self, iteration: int) -> dict[str, float]:
        noises = self.rng.standard_normal((self.population_size, self.policy.num_params), dtype=np.float32)
        rewards = np.empty((self.population_size, 2), dtype=np.float32)
        lengths = np.empty((self.population_size, 2), dtype=np.int32)
        env = self.env_factory()
        try:
            for index, noise in enumerate(noises):
                episode_seed = int(self.rng.integers(0, 2**31 - 1))
                for sign_index, sign in enumerate((1.0, -1.0)):
                    candidate = self.policy.parameters + sign * self.noise_std * noise
                    collect = self.rng.random() < self.observation_update_probability
                    result = self.policy.rollout(env, candidate, seed=episode_seed,
                                                 action_noise_std=self.action_noise_std,
                                                 collect_observations=collect)
                    rewards[index, sign_index], lengths[index, sign_index] = result[:2]
                    if collect:
                        self._accumulate_observation_statistics(result[2])
        finally:
            env.close()
        self._apply_observation_statistics()
        shaped = centered_ranks(rewards)
        gradient = noises.T @ (shaped[:, 0] - shaped[:, 1])
        gradient /= self.population_size * self.noise_std
        gradient -= self.l2_coefficient * self.policy.parameters
        self.policy.parameters += self.optimizer.step(gradient).astype(np.float32)
        self.episodes += rewards.size
        return {
            "iteration": float(iteration), "reward_mean": float(rewards.mean()),
            "reward_max": float(rewards.max()), "length_mean": float(lengths.mean()),
            "episodes": float(self.episodes),
        }

    def evaluate(self, episodes: int = 5, seed: int = 0, render_mode: str | None = None) -> float:
        env = self.env_factory(render_mode=render_mode)
        try:
            returns = [self.policy.rollout(env, seed=seed + index, render=render_mode == "human")[0]
                       for index in range(episodes)]
        finally:
            env.close()
        return float(np.mean(returns))
