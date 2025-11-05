#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: T-Rex Dino Bot (Reinforcement Learning)
File: models.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-05
Updated: 2025-11-05
License: MIT License (see LICENSE file for details)
=

Description:
Neural network models for the Dino DQN agent. Currently provides a simple
multi-layer perceptron (MLP) for approximating the Q-function.

Usage:
from trex_dino_bot.models import DQNNetwork

model = DQNNetwork(state_dim=5, action_dim=2)
q_values = model(states)

Notes:
- The architecture is intentionally small to allow training on CPU or
  modest GPU (e.g., GTX 1050).

=================================================================================================================
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNNetwork(nn.Module):
    """Simple fully-connected DQN network."""

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        hidden = 128
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, action_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
