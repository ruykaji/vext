"""Extensible command-line runner for MuJoCo and Atari ES experiments."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from types import ModuleType

from es_distributed import EvolutionStrategy, MLPPolicy

ROOT = Path(__file__).resolve().parent


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    for key in ("backend", "env_id", "policy", "training"):
        if key not in config:
            raise ValueError(f"configuration is missing {key!r}")
    return config


def load_backend(name: str) -> ModuleType:
    path = ROOT / name / "environment.py"
    if not path.is_file():
        raise ValueError(f"unknown backend {name!r}; expected {path}")
    spec = importlib.util.spec_from_file_location(f"es_backend_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load backend from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "make_env"):
        raise TypeError(
            f"backend {name!r} must define make_env(config, render_mode=None)"
        )
    return module


def build(config: dict) -> tuple[MLPPolicy, EvolutionStrategy]:
    backend = load_backend(config["backend"])
    env_factory = lambda render_mode=None: backend.make_env(
        config, render_mode=render_mode
    )
    env = env_factory()
    try:
        policy = MLPPolicy(
            env.observation_space,
            env.action_space,
            tuple(config["policy"].get("hidden_sizes", [32, 32])),
            config.get("seed", 0),
        )
    finally:
        env.close()
    training = config["training"]
    trainer = EvolutionStrategy(
        env_factory,
        policy,
        population_size=training["population_size"],
        noise_std=training["noise_std"],
        learning_rate=training["learning_rate"],
        l2_coefficient=training.get("l2_coefficient", 0.005),
        action_noise_std=training.get("action_noise_std", 0.0),
        observation_update_probability=training.get(
            "observation_update_probability", 0.0
        ),
        seed=config.get("seed", 0),
    )
    return policy, trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MuJoCo or Atari policies with OpenAI-style ES"
    )
    parser.add_argument("config", type=Path, help="JSON experiment configuration")
    parser.add_argument("--iterations", type=int, help="override configured iterations")
    parser.add_argument(
        "--population-size", type=int, help="override perturbation pairs per iteration"
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/policy.npz"))
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.population_size is not None:
        config["training"]["population_size"] = args.population_size
    policy, trainer = build(config)
    if args.evaluate:
        policy.load(str(args.checkpoint))
        score = trainer.evaluate(
            config.get("evaluation_episodes", 5),
            seed=config.get("seed", 0),
            render_mode="human" if args.render else None,
        )
        print(f"mean_return={score:.3f}")
        return

    iterations = (
        args.iterations
        if args.iterations is not None
        else config["training"]["iterations"]
    )
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = args.checkpoint.parent / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = None
        for iteration in range(1, iterations + 1):
            metrics = trainer.train_iteration(iteration)
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=metrics)
                writer.writeheader()
            writer.writerow(metrics)
            stream.flush()
            print(
                "iteration={iteration:.0f} reward_mean={reward_mean:.3f} reward_max={reward_max:.3f} "
                "length_mean={length_mean:.1f}".format(**metrics),
                flush=True,
            )
    policy.save(str(args.checkpoint))
    score = trainer.evaluate(
        config.get("evaluation_episodes", 5), seed=config.get("seed", 0)
    )
    print(f"saved={args.checkpoint} evaluation_mean={score:.3f}")


if __name__ == "__main__":
    main()
