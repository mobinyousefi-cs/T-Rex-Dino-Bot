#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: T-Rex Dino Bot (Reinforcement Learning)
File: __init__.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-05
Updated: 2025-11-05
License: MIT License (see LICENSE file for details)
=

Description:
Package initialization for the T-Rex Dino Bot project. Exposes the main
public API surface such as the DinoEnv and DQNAgent classes.

Usage:
from trex_dino_bot import DinoEnv, DQNAgent

Notes:
- See config.py for hyper-parameters.
- main.py (in canvas) contains the CLI entry point.

=================================================================================================================
"""

from .config import *  # noqa: F401, F403
from .env import DinoEnv  # noqa: F401
from .agent import DQNAgent  # noqa: F401

__all__ = [
    "DinoEnv",
    "DQNAgent",
]
