#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: T-Rex Dino Bot (Reinforcement Learning)
File: main.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-05
Updated: 2025-11-05
License: MIT License (see LICENSE file for details)
=

Description:
Command-line entry point for training and evaluating a DQN-based agent
that learns to play a simplified T-Rex Dino game implemented with
pygame. The script wires together configuration, environment, and agent
components and exposes a simple CLI.

Usage:
python -m trex_dino_bot.main --mode train --episodes 1000
python -m trex_dino_bot.main --mode play --checkpoint checkpoints/dqn_latest.pt

Notes:
- Designed to be lightweight and easy to extend.
- Training is stochastic; use config.RANDOM_SEED for reproducibility.

=================================================================================================================
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import torch

from . import config
from .agent import DQNAgent
from .env import DinoEnv


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with fields ``mode``, ``episodes``, ``checkpoint``,
        and ``render``.
    """

    parser = argparse.ArgumentParser(
        description="Train or evaluate a DQN agent on the T-Rex Dino environment.",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "play"],
        default="train",
        help="Train a new agent or play using a trained checkpoint.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Number of episodes to train or play for.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/dqn_latest.pt",
        help="Path to save/load the agent checkpoint.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment during training or play.",
    )
    return parser.parse_args()


def ensure_checkpoint_dir(path: str | os.PathLike[str]) -> None:
    """Ensure the parent directory of a checkpoint path exists."""

    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)


def train(episodes: int, checkpoint_path: str, render: bool) -> None:
    """Train the DQN agent on the Dino environment.

    Parameters
    ----------
    episodes:
        Number of training episodes.
    checkpoint_path:
        File path to which the trained model weights will be saved.
    render:
        Whether to render the environment during training.
    """

    env = DinoEnv(
        width=config.SCREEN_WIDTH,
        height=config.SCREEN_HEIGHT,
        enable_render=render,
    )

    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        cfg=config,
    )

    ensure_checkpoint_dir(checkpoint_path)

    best_mean_reward: Optional[float] = None
    reward_history: list[float] = []

    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()

            state = next_state
            episode_reward += reward

            if render:
                env.render()

        reward_history.append(episode_reward)

        if episode % config.LOG_INTERVAL == 0:
            mean_reward = sum(reward_history[-config.LOG_INTERVAL :]) / float(
                config.LOG_INTERVAL
            )
            best_mean_reward = (
                mean_reward
                if best_mean_reward is None
                else max(best_mean_reward, mean_reward)
            )
            print(
                f"[Episode {episode:04d}] Reward: {episode_reward:.1f} | "
                f"Mean({config.LOG_INTERVAL}) = {mean_reward:.1f} | "
                f"Best Mean = {best_mean_reward:.1f}",
            )

        if episode % config.CHECKPOINT_INTERVAL == 0:
            torch.save(agent.q_net.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    # Save final weights
    torch.save(agent.q_net.state_dict(), checkpoint_path)
    print(f"Training completed. Final checkpoint saved to {checkpoint_path}")

    env.close()


def play(episodes: int, checkpoint_path: str, render: bool) -> None:
    """Play episodes using a trained agent.

    Parameters
    ----------
    episodes:
        Number of evaluation episodes.
    checkpoint_path:
        Path to the trained model weights.
    render:
        Whether to render the environment.
    """

    if not render:
        # For play mode we generally want to see the agent.
        render = True

    env = DinoEnv(
        width=config.SCREEN_WIDTH,
        height=config.SCREEN_HEIGHT,
        enable_render=render,
    )

    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        cfg=config,
        eval_mode=True,
    )

    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint '{checkpoint_path}' does not exist. "
            "Train an agent first using --mode train.",
        )

    agent.q_net.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE))
    agent.q_net.eval()

    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            action = agent.select_action(state, exploit_only=True)
            next_state, reward, done, info = env.step(action)
            state = next_state
            episode_reward += reward

            if render:
                env.render()

        print(f"[PLAY] Episode {episode:04d} reward: {episode_reward:.1f}")

    env.close()


def main() -> None:
    """Entry point for the CLI script."""

    args = parse_args()

    if args.mode == "train":
        train(episodes=args.episodes, checkpoint_path=args.checkpoint, render=args.render)
    else:
        play(episodes=args.episodes, checkpoint_path=args.checkpoint, render=args.render)


if __name__ == "__main__":
    main()
