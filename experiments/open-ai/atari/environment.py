"""Atari/ALE adapter with standard frame preprocessing."""

import ale_py
import gymnasium as gym
from gymnasium.wrappers import (
    AtariPreprocessing,
    FlattenObservation,
    FrameStackObservation,
)


def make_env(config: dict, render_mode: str | None = None) -> gym.Env:
    gym.register_envs(ale_py)
    options = config.get("environment", {})
    env = gym.make(
        config["env_id"],
        render_mode=render_mode,
        frameskip=1,
        repeat_action_probability=options.get("repeat_action_probability", 0.0),
    )
    env = AtariPreprocessing(
        env,
        noop_max=options.get("noop_max", 30),
        frame_skip=options.get("frame_skip", 4),
        screen_size=options.get("screen_size", 42),
        terminal_on_life_loss=options.get("terminal_on_life_loss", False),
        grayscale_obs=options.get("grayscale_obs", True),
        scale_obs=False,
    )
    env = FrameStackObservation(env, stack_size=options.get("frame_stack", 4))
    return FlattenObservation(env)
