#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: T-Rex Dino Bot (Reinforcement Learning)
File: config.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-05
Updated: 2025-11-05
License: MIT License (see LICENSE file for details)
=

Description:
Central configuration and hyper-parameters for the Dino DQN agent and
environment.

Usage:
from trex_dino_bot import config

Notes:
- Adjust hyper-parameters here for experiments.
- DEVICE is automatically resolved to 'cuda' if available.

=================================================================================================================
"""

from __future__ import annotations

import torch

# Reproducibility
RANDOM_SEED: int = 42

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Screen / environment
SCREEN_WIDTH: int = 800
SCREEN_HEIGHT: int = 200
GROUND_Y: int = 150
FPS: int = 60

# Dino parameters
DINO_WIDTH: int = 40
DINO_HEIGHT: int = 40
DINO_X: int = 80
JUMP_VELOCITY: float = -14.0
GRAVITY: float = 1.0

# Obstacle parameters
OBSTACLE_WIDTH: int = 20
OBSTACLE_HEIGHT: int = 40
OBSTACLE_SPEED: float = 8.0
OBSTACLE_MIN_GAP: int = 200
OBSTACLE_MAX_GAP: int = 350

# RL parameters
GAMMA: float = 0.99
LR: float = 1e-3
BATCH_SIZE: int = 64
REPLAY_CAPACITY: int = 50_000
MIN_REPLAY_SIZE: int = 5_000

EPS_START: float = 1.0
EPS_END: float = 0.05
EPS_DECAY_STEPS: int = 100_000

TARGET_UPDATE_INTERVAL: int = 1_000
CHECKPOINT_INTERVAL: int = 100
LOG_INTERVAL: int = 10

MAX_STEPS_PER_EPISODE: int = 10_000
