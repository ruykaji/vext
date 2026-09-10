"""Small NumPy policies for Gymnasium environments."""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class Layer:
    weight: slice
    bias: slice
    input_dim: int
    output_dim: int


class MLPPolicy:
    """A flat-parameter MLP supporting Box and Discrete action spaces."""

    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space,
                 hidden_sizes: tuple[int, ...] = (32, 32), seed: int = 0):
        if not isinstance(observation_space, gym.spaces.Box) or len(observation_space.shape) != 1:
            raise TypeError("Only one-dimensional Box observation spaces are supported")
        if isinstance(action_space, gym.spaces.Box):
            if len(action_space.shape) != 1:
                raise TypeError("Only one-dimensional Box action spaces are supported")
            output_dim = int(action_space.shape[0])
        elif isinstance(action_space, gym.spaces.Discrete):
            output_dim = int(action_space.n)
        else:
            raise TypeError("Action space must be Box or Discrete")

        self.action_space = action_space
        self.observation_mean = np.zeros(observation_space.shape, dtype=np.float32)
        self.observation_std = np.ones(observation_space.shape, dtype=np.float32)
        dims = (int(observation_space.shape[0]), *hidden_sizes, output_dim)
        self.layers: list[Layer] = []
        offset = 0
        for input_dim, layer_output_dim in zip(dims, dims[1:]):
            weight_count = input_dim * layer_output_dim
            weight = slice(offset, offset + weight_count)
            offset += weight_count
            bias = slice(offset, offset + layer_output_dim)
            offset += layer_output_dim
            self.layers.append(Layer(weight, bias, input_dim, layer_output_dim))
        self.num_params = offset

        rng = np.random.default_rng(seed)
        self.parameters = np.zeros(self.num_params, dtype=np.float32)
        for layer in self.layers:
            weights = rng.normal(size=(layer.input_dim, layer.output_dim)).astype(np.float32)
            weights /= np.sqrt(np.square(weights).sum(axis=0, keepdims=True)).clip(min=1e-8)
            self.parameters[layer.weight] = weights.ravel()

    def act(self, observation: np.ndarray, parameters: np.ndarray | None = None,
            action_noise: np.ndarray | None = None):
        params = self.parameters if parameters is None else parameters
        x = np.clip((np.asarray(observation, dtype=np.float32) - self.observation_mean)
                    / self.observation_std, -5.0, 5.0)
        for index, layer in enumerate(self.layers):
            weights = params[layer.weight].reshape(layer.input_dim, layer.output_dim)
            x = x @ weights + params[layer.bias]
            if index < len(self.layers) - 1:
                x = np.tanh(x)
        if isinstance(self.action_space, gym.spaces.Discrete):
            return int(np.argmax(x))
        low = self.action_space.low.astype(np.float32)
        high = self.action_space.high.astype(np.float32)
        action = low + (np.tanh(x) + 1.0) * 0.5 * (high - low)
        if action_noise is not None:
            action = np.clip(action + action_noise, low, high)
        return action

    def rollout(self, env: gym.Env, parameters: np.ndarray | None = None, *, seed: int | None = None,
                render: bool = False, action_noise_std: float = 0.0,
                collect_observations: bool = False):
        observation, _ = env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        total_reward = 0.0
        length = 0
        observations = []
        while True:
            if collect_observations:
                observations.append(observation)
            action_noise = None
            if action_noise_std and isinstance(self.action_space, gym.spaces.Box):
                action_noise = rng.normal(0.0, action_noise_std, self.action_space.shape)
            observation, reward, terminated, truncated, _ = env.step(
                self.act(observation, parameters, action_noise)
            )
            total_reward += float(reward)
            length += 1
            if render:
                env.render()
            if terminated or truncated:
                result = (total_reward, length)
                if collect_observations:
                    return (*result, np.asarray(observations, dtype=np.float32))
                return result

    def save(self, path: str) -> None:
        np.savez_compressed(path, parameters=self.parameters,
                            observation_mean=self.observation_mean, observation_std=self.observation_std)

    def load(self, path: str) -> None:
        with np.load(path) as data:
            parameters = np.asarray(data["parameters"], dtype=np.float32)
            observation_mean = np.asarray(data.get("observation_mean", self.observation_mean), dtype=np.float32)
            observation_std = np.asarray(data.get("observation_std", self.observation_std), dtype=np.float32)
        if parameters.shape != self.parameters.shape:
            raise ValueError(f"checkpoint has {parameters.size} parameters; expected {self.num_params}")
        self.parameters[:] = parameters
        self.observation_mean[:] = observation_mean
        self.observation_std[:] = observation_std
