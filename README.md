# T-Rex Dino Bot 🦖

A clean, modular **Reinforcement Learning** project that trains a Deep Q-Network
(DQN) agent to play a simplified version of the famous **Chrome T-Rex Dino game**
implemented with `pygame`.

Author: **Mobin Yousefi** ([GitHub: mobinyousefi-cs](https://github.com/mobinyousefi-cs))  
License: **MIT**

---

## Features

- Custom **pygame** environment (`DinoEnv`) with a Gym-like API
- **DQN agent** with:
  - Experience replay
  - Target network
  - Epsilon-greedy exploration with linear decay
- Lightweight **MLP** network suitable for CPU and modest GPUs (e.g., GTX 1050)
- Clean `src/` layout, tests, and GitHub Actions CI
- Designed as a learning-friendly RL project you can extend with more advanced ideas
  (Double DQN, Dueling networks, Prioritized Replay, etc.)

---

## Installation

```bash
git clone https://github.com/mobinyousefi-cs/trex-dino-bot.git
cd trex-dino-bot

# (optional) create virtual environment
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\\Scripts\\activate

# install package
pip install -e .
