#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: T-Rex Dino Bot (Reinforcement Learning)
File: env.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-05
Updated: 2025-11-05
License: MIT License (see LICENSE file for details)
=

Description:
Simplified T-Rex Dino environment implemented with pygame. Provides a
Gym-like API (reset, step, render, close) and returns a compact
feature-based observation vector suitable for value-based RL methods.

Usage:
from trex_dino_bot.env import DinoEnv

env = DinoEnv(width=800, height=200, enable_render=True)
state = env.reset()
for t in range(1000):
    action = env.sample_action()
    state, reward, done, info = env.step(action)
    env.render()
    if done:
        break

Notes:
- This is not a pixel-based environment; state is a low-dimensional vector
  [normalized distance, obstacle_width, obstacle_height, vertical_velocity,
   is_on_ground].

=================================================================================================================
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, Tuple

import numpy as np
import pygame

from . import config


class DinoEnv:
    """Minimal side-scrolling Dino environment."""

    def __init__(self, width: int, height: int, enable_render: bool = False) -> None:
        self.width = width
        self.height = height
        self.enable_render = enable_render

        self._screen: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None

        self.state_dim: int = 5
        self.action_dim: int = 2  # 0: do nothing, 1: jump

        self.dino_rect = pygame.Rect(
            config.DINO_X,
            config.GROUND_Y - config.DINO_HEIGHT,
            config.DINO_WIDTH,
            config.DINO_HEIGHT,
        )
        self.dino_vel_y: float = 0.0

        self.obstacle_rect = pygame.Rect(0, 0, config.OBSTACLE_WIDTH, config.OBSTACLE_HEIGHT)
        self.obstacle_speed: float = config.OBSTACLE_SPEED
        self._next_obstacle_x: int = 0

        self.timestep: int = 0
        self._rng = random.Random(config.RANDOM_SEED)

        if self.enable_render:
            self._init_pygame()

    # ------------------------------------------------------------------ #
    # Core API                                                           #
    # ------------------------------------------------------------------ #

    def reset(self) -> np.ndarray:
        """Reset the environment state and return initial observation."""

        self.timestep = 0
        self.dino_rect.y = config.GROUND_Y - config.DINO_HEIGHT
        self.dino_vel_y = 0.0

        self._spawn_obstacle()

        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Advance one step in the environment.

        Parameters
        ----------
        action:
            0: do nothing, 1: jump.

        Returns
        -------
        state: np.ndarray
        reward: float
        done: bool
        info: dict
        """

        self.timestep += 1

        # Apply action
        if action == 1 and self._is_on_ground:
            self.dino_vel_y = config.JUMP_VELOCITY

        # Physics integration
        self.dino_vel_y += config.GRAVITY
        self.dino_rect.y += int(self.dino_vel_y)

        # Clamp to ground
        if self.dino_rect.y >= config.GROUND_Y - config.DINO_HEIGHT:
            self.dino_rect.y = config.GROUND_Y - config.DINO_HEIGHT
            self.dino_vel_y = 0.0

        # Move obstacle
        self.obstacle_rect.x -= int(self.obstacle_speed)
        if self.obstacle_rect.right < 0:
            self._spawn_obstacle()

        # Collision detection
        done = self.dino_rect.colliderect(self.obstacle_rect)
        reward = -1.0 if done else 1.0

        # Time limit
        if self.timestep >= config.MAX_STEPS_PER_EPISODE:
            done = True

        state = self._get_state()
        info: Dict[str, Any] = {"timestep": self.timestep}

        return state, reward, done, info

    def render(self) -> None:
        """Render current state using pygame."""
        if not self.enable_render:
            return

        if self._screen is None or self._clock is None:
            self._init_pygame()

        assert self._screen is not None
        assert self._clock is not None

        self._screen.fill((255, 255, 255))
        # Ground line
        pygame.draw.line(
            self._screen,
            (0, 0, 0),
            (0, config.GROUND_Y),
            (self.width, config.GROUND_Y),
            2,
        )

        # Dino
        pygame.draw.rect(self._screen, (0, 0, 0), self.dino_rect)

        # Obstacle
        pygame.draw.rect(self._screen, (255, 0, 0), self.obstacle_rect)

        pygame.display.flip()
        self._clock.tick(config.FPS)

    def close(self) -> None:
        """Clean up pygame resources."""
        if self._screen is not None:
            pygame.display.quit()
        if self._clock is not None:
            pygame.quit()

        self._screen = None
        self._clock = None

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    @property
    def _is_on_ground(self) -> bool:
        return self.dino_rect.y >= config.GROUND_Y - config.DINO_HEIGHT - 1

    def _spawn_obstacle(self) -> None:
        gap = self._rng.randint(config.OBSTACLE_MIN_GAP, config.OBSTACLE_MAX_GAP)
        x = self.width + gap
        self.obstacle_rect.width = config.OBSTACLE_WIDTH
        self.obstacle_rect.height = config.OBSTACLE_HEIGHT
        self.obstacle_rect.x = x
        self.obstacle_rect.y = config.GROUND_Y - config.OBSTACLE_HEIGHT

    def _get_state(self) -> np.ndarray:
        distance = self.obstacle_rect.x - self.dino_rect.x
        distance_norm = max(distance / float(self.width), 0.0)
        vel_norm = self.dino_vel_y / 20.0

        is_on_ground = 1.0 if self._is_on_ground else 0.0

        state = np.array(
            [
                distance_norm,
                self.obstacle_rect.width / float(self.width),
                self.obstacle_rect.height / float(self.height),
                vel_norm,
                is_on_ground,
            ],
            dtype=np.float32,
        )
        return state

    def sample_action(self) -> int:
        """Sample a random action."""
        return self._rng.randint(0, self.action_dim - 1)

    # ------------------------------------------------------------------ #
    # Initialization                                                     #
    # ------------------------------------------------------------------ #

    def _init_pygame(self) -> None:
        pygame.init()
        self._screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("T-Rex Dino Bot")
        self._clock = pygame.time.Clock()
