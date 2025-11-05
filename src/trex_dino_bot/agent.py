#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: T-Rex Dino Bot (Reinforcement Learning)
File: agent.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-05
Updated: 2025-11-05
License: MIT License (see LICENSE file for details)
=

Description:
Deep Q-Network (DQN) agent implementation for the Dino environment,
including epsilon-greedy exploration, target network updates, and
experience replay.

Usage:
from trex_dino_bot.agent import DQNAgent
from trex_dino_bot.env import DinoEnv
from trex_dino_bot import config

env = DinoEnv(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
agent = DQNAgent(env.state_dim, env.action_dim, cfg=config)

state = env.reset()
for t in range(1000):
    action = agent.select_action(state)
    next_state, reward, done, _ = env.step(action)
    agent.store_transition(state, action, reward, next_state, done)
    agent.update()
    state = next_state
    if done:
        break

Notes:
- Designed to be simple and readable rather than aggressively optimized.

=================================================================================================================
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from . import config
from .models import DQNNetwork
from .replay_buffer import ReplayBuffer


class DQNAgent:
    """Deep Q-Network agent for DinoEnv."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        cfg: Any = config,
        eval_mode: bool = False,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.cfg = cfg
        self.eval_mode = eval_mode

        self.device = cfg.DEVICE

        self.q_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg.LR)
        self.replay = ReplayBuffer(cfg.REPLAY_CAPACITY, state_dim)

        self.steps_done: int = 0

    # ------------------------------------------------------------------ #
    # Interaction                                                        #
    # ------------------------------------------------------------------ #

    def select_action(
        self,
        state: np.ndarray,
        exploit_only: bool = False,
    ) -> int:
        """Select an action using epsilon-greedy exploration."""
        eps = 0.0 if exploit_only or self.eval_mode else self._epsilon

        if np.random.random() < eps:
            return np.random.randint(self.action_dim)

        state_t = torch.from_numpy(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        action = int(q_values.argmax(dim=1).item())
        return action

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.replay.push(state, action, reward, next_state, done)
        self.steps_done += 1

    # ------------------------------------------------------------------ #
    # Learning                                                           #
    # ------------------------------------------------------------------ #

    @property
    def _epsilon(self) -> float:
        """Linearly decaying epsilon schedule."""
        frac = min(self.steps_done / float(self.cfg.EPS_DECAY_STEPS), 1.0)
        return self.cfg.EPS_START + frac * (self.cfg.EPS_END - self.cfg.EPS_START)

    def update(self) -> None:
        """Run one DQN update step if there are enough samples."""
        if self.replay.size < self.cfg.MIN_REPLAY_SIZE:
            return

        batch = self.replay.sample(self.cfg.BATCH_SIZE, self.device)

        # Q(s, a)
        q_values = self.q_net(batch.states)
        state_action_values = q_values.gather(1, batch.actions.unsqueeze(1)).squeeze(1)

        # Target: r + gamma * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values = self.target_net(batch.next_states)
            max_next_q_values = next_q_values.max(dim=1)[0]
            target_values = batch.rewards + (1.0 - batch.dones) * self.cfg.GAMMA * max_next_q_values

        loss = nn.functional.mse_loss(state_action_values, target_values)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Periodically update target network
        if self.steps_done % self.cfg.TARGET_UPDATE_INTERVAL == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
