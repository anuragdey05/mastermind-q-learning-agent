"""
Provided Q-table helper for the Mastermind assignment.

Students are not expected to modify this file. They only need to use the
public API:
- `get(state, action)`
- `set(state, action, value)`
- `get_max_value(state)`
- `get_best_action(state)`
"""

from typing import Dict, Optional, Tuple

import numpy as np


State = Tuple


class QTable:
    """
    Tabular Q-function with the usual Q(state, action) interface.

    Internally, each visited state stores a NumPy vector of action values.
    This keeps the interface simple while avoiding a huge dense table for every
    possible Mastermind state.

    Unseen state-action values default to `0.0`.
    Greedy ties are broken randomly.
    """

    def __init__(self, num_actions: int, initial_value: float = 0.0):
        self.num_actions = num_actions
        self.initial_value = initial_value
        self.q: Dict[State, np.ndarray] = {}

    def _get_action_values(self, state: State, create: bool = False) -> Optional[np.ndarray]:
        """Return the action-value vector for one state."""
        if state in self.q:
            return self.q[state]

        if not create:
            return None

        self.q[state] = np.full(self.num_actions, self.initial_value, dtype=float)
        return self.q[state]

    def get(self, state: State, action: int) -> float:
        """Return Q(state, action)."""
        action_values = self._get_action_values(state, create=False)
        if action_values is None:
            return self.initial_value
        return float(action_values[action])

    def set(self, state: State, action: int, value: float):
        """Store Q(state, action)."""
        action_values = self._get_action_values(state, create=True)
        if action_values is None:
            return
        action_values[action] = value

    def get_max_value(self, state: State) -> float:
        """Return max_a Q(state, a)."""
        action_values = self._get_action_values(state, create=False)
        if action_values is None:
            return self.initial_value
        return float(np.max(action_values))

    def get_best_action(self, state: State) -> int:
        """Return a greedy action, breaking ties randomly."""
        action_values = self._get_action_values(state, create=False)
        if action_values is None:
            return int(np.random.randint(self.num_actions))

        max_value = np.max(action_values)
        best_actions = np.flatnonzero(action_values == max_value)
        return int(np.random.choice(best_actions))
