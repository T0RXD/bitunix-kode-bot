"""
Q-Learning Reinforcement Learning Agent for Bitunix Futures Bot

State representation:
  A tuple of 4 binary indicators: (EMA_cross, BB_breakout, Vol_spike, PSAR_pos)
  Each element is 0 or 1, giving 2^4 = 16 possible states.

Actions:
  0 = No Action (skip trade)
  1 = Enter Long

The agent maintains a Q-Table (states x actions) and updates it using
the standard Q-Learning update rule:
  Q(s, a) <- Q(s, a) + alpha * (reward + gamma * max_a' Q(s', a') - Q(s, a))

Persistence:
  The Q-Table is saved to / loaded from 'q_table.npy'.
"""

import logging
import os
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# State is a tuple of 4 binary values
StateTuple = Tuple[int, int, int, int]

NUM_INDICATORS = 4
NUM_STATES = 2 ** NUM_INDICATORS  # 16
NUM_ACTIONS = 2  # 0=No Action, 1=Enter Long

ACTION_NO_OP = 0
ACTION_ENTER_LONG = 1


def state_to_index(state: StateTuple) -> int:
    """Convert a binary state tuple to an integer index (0..15)."""
    idx = 0
    for bit in state:
        idx = (idx << 1) | int(bool(bit))
    return idx


def index_to_state(idx: int) -> StateTuple:
    """Convert an integer index (0..15) back to a binary state tuple."""
    return (
        (idx >> 3) & 1,
        (idx >> 2) & 1,
        (idx >> 1) & 1,
        idx & 1,
    )


class QLearningAgent:
    """
    Tabular Q-Learning agent for trade entry decisions.

    Parameters
    ----------
    alpha : float
        Learning rate (default 0.1).
    gamma : float
        Discount factor (default 0.95).
    epsilon : float
        Exploration rate for epsilon-greedy policy (default 0.1).
    q_table_path : str
        File path for persisting the Q-Table.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        q_table_path: str = "q_table.npy",
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table_path = q_table_path
        self.q_table: np.ndarray = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=np.float64)
        self._load()

    def _load(self) -> None:
        """Load Q-Table from disk if it exists."""
        if os.path.isfile(self.q_table_path):
            try:
                loaded = np.load(self.q_table_path)
                if loaded.shape == (NUM_STATES, NUM_ACTIONS):
                    self.q_table = loaded.astype(np.float64)
                    logger.info("Loaded Q-Table from %s", self.q_table_path)
                else:
                    logger.warning(
                        "Q-Table shape mismatch (%s vs expected %s), starting fresh",
                        loaded.shape,
                        (NUM_STATES, NUM_ACTIONS),
                    )
            except Exception as e:
                logger.warning("Failed to load Q-Table from %s: %s", self.q_table_path, e)

    def save(self) -> None:
        """Persist Q-Table to disk."""
        try:
            np.save(self.q_table_path, self.q_table)
            logger.debug("Saved Q-Table to %s", self.q_table_path)
        except Exception as e:
            logger.error("Failed to save Q-Table: %s", e)

    def get_action(self, state: StateTuple, epsilon: Optional[float] = None) -> int:
        """
        Choose an action using epsilon-greedy policy.

        Parameters
        ----------
        state : StateTuple
            Current market state as (EMA_cross, BB_breakout, Vol_spike, PSAR_pos).
        epsilon : float, optional
            Override exploration rate for this call. Uses self.epsilon if None.

        Returns
        -------
        int
            0 (No Action) or 1 (Enter Long).
        """
        eps = epsilon if epsilon is not None else self.epsilon
        idx = state_to_index(state)

        if np.random.random() < eps:
            return int(np.random.randint(NUM_ACTIONS))

        q_values = self.q_table[idx]
        # Break ties randomly
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        return int(np.random.choice(best_actions))

    def update(
        self,
        state: StateTuple,
        action: int,
        reward: float,
        next_state: StateTuple,
    ) -> None:
        """
        Update the Q-Table using the standard Q-Learning formula.

        Q(s,a) <- Q(s,a) + alpha * (reward + gamma * max_a' Q(s',a') - Q(s,a))
        """
        s = state_to_index(state)
        s_next = state_to_index(next_state)
        a = int(action)

        current_q = self.q_table[s, a]
        max_next_q = np.max(self.q_table[s_next])

        td_target = reward + self.gamma * max_next_q
        self.q_table[s, a] = current_q + self.alpha * (td_target - current_q)

        self.save()

    def get_q_values(self, state: StateTuple) -> np.ndarray:
        """Return Q-values for all actions in the given state."""
        return self.q_table[state_to_index(state)].copy()

    def reset(self) -> None:
        """Reset Q-Table to zeros."""
        self.q_table = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=np.float64)
        self.save()
