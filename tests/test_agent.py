#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: T-Rex Dino Bot (Reinforcement Learning)
File: test_agent.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-05
Updated: 2025-11-05
License: MIT License (see LICENSE file for details)
=

Description:
Basic smoke tests for the DQNAgent to ensure interaction with DinoEnv
and replay buffer works as expected.

Usage:
pytest tests/test_agent.py

Notes:
- Does not test learning performance, only shapes and integration.

=================================================================================================================
"""

from __future__ import annotations

import numpy as np

from trex_dino_bot import DinoEnv, DQNAgent, config


def test_agent_step_and_update() -> None:
    env = DinoEnv(width=config.SCREEN_WIDTH, height=config.SCREEN_HEIGHT, enable_render=False)
    agent = DQNAgent(env.state_dim, env.action_dim, cfg=config)

    state = env.reset()
    for _ in range(config.BATCH_SIZE + 5):
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        agent.update()
        state = next_state if not done else env.reset()

    env.close()
