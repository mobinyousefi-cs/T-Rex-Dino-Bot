#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: T-Rex Dino Bot (Reinforcement Learning)
File: test_env.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-05
Updated: 2025-11-05
License: MIT License (see LICENSE file for details)
=

Description:
Basic smoke tests for the DinoEnv environment.

Usage:
pytest tests/test_env.py

Notes:
- Designed as lightweight sanity checks for CI.

=================================================================================================================
"""

from __future__ import annotations

from trex_dino_bot import DinoEnv, config


def test_env_reset_and_step() -> None:
    env = DinoEnv(width=config.SCREEN_WIDTH, height=config.SCREEN_HEIGHT, enable_render=False)
    state = env.reset()
    assert state.shape[0] == env.state_dim

    for _ in range(10):
        action = env.sample_action()
        next_state, reward, done, info = env.step(action)
        assert next_state.shape[0] == env.state_dim
        assert isinstance(reward, float)
        assert isinstance(done, bool)
    env.close()
