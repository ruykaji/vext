"""MuJoCo environment adapter. This directory intentionally has no __init__.py."""

import gymnasium as gym


def make_env(config: dict, render_mode: str | None = None) -> gym.Env:
    return gym.make(
        config["env_id"], render_mode=render_mode, **config.get("environment", {})
    )
