"""
Mastermind Environment
======================

Game Setup (configurable):
- num_positions: positions in the code (default 2)
- num_colors: colors available (default 4, reduce to 3 if slow)
- max_turns: maximum guesses allowed (default 6)
- history_length: how many previous turns in state (default 4)

State Representation:
Each turn record stores all guess positions followed by `(black, white)`.
The state is a tuple of the last N turn records.
Empty slots (turns not yet played) are represented as None.

Note: With 2 positions and 4 colors, there are 16 possible codes.
      With 2 positions and 3 colors, there are 9 possible codes (faster).
"""

import numpy as np
from itertools import product
from typing import Tuple, List, Optional


class MastermindEnv:
    """Mastermind environment with discrete actions and episode state."""

    def __init__(
        self,
        num_positions: int = 2,
        num_colors: int = 4,
        max_turns: int = 6,
        history_length: int = 4,
    ):
        """
        Args:
            num_positions (int): Number of positions in the code.
            num_colors (int): Number of available colors.
            max_turns (int): Maximum number of guesses in one episode.
            history_length (int): Number of recent turns stored in the state.
        """
        self.num_positions = num_positions
        self.num_colors = num_colors
        self.max_turns = max_turns
        self.history_length = history_length
        
        # All possible codes
        self.all_codes = list(product(range(num_colors), repeat=num_positions))
        self.num_codes = len(self.all_codes)
        
        # Episode state
        self.secret: Optional[Tuple[int, ...]] = None
        self.turn: int = 0
        self.history: List[Tuple[Tuple[int, ...], int, int]] = []  # (guess, black, white)
        self.done: bool = False
    
    def compute_feedback(self, guess: Tuple[int, ...], secret: Tuple[int, ...]) -> Tuple[int, int]:
        """
        Compute black and white peg counts.

        Args:
            guess (tuple): Guessed code.
            secret (tuple): Secret code.

        Returns:
            tuple: (black, white), where black counts exact matches and white
            counts correct colors in the wrong position.
        """
        # Count black (exact matches)
        black = sum(g == s for g, s in zip(guess, secret))
        
        # Count total color matches
        guess_counts = {}
        secret_counts = {}
        for g, s in zip(guess, secret):
            guess_counts[g] = guess_counts.get(g, 0) + 1
            secret_counts[s] = secret_counts.get(s, 0) + 1
        
        total_matches = sum(
            min(guess_counts.get(c, 0), secret_counts.get(c, 0))
            for c in set(guess_counts) | set(secret_counts)
        )
        
        # White = total matches - black
        white = total_matches - black
        
        return black, white
    
    def action_to_code(self, action: int) -> Tuple[int, ...]:
        """
        Convert an action index to a code tuple.

        Args:
            action (int): Integer action index.

        Returns:
            tuple: Code corresponding to that action.
        """
        return self.all_codes[action]
    
    def code_to_action(self, code: Tuple[int, ...]) -> int:
        """
        Convert a code tuple to an action index.

        Args:
            code (tuple): Code tuple.

        Returns:
            int: Integer action index.
        """
        return self.all_codes.index(code)
    
    def get_state(self) -> Tuple:
        """
        Return the current state as a tuple of recent turn records.

        Each record has the form `(*guess, black, white)`, where `guess`
        contributes `num_positions` entries.
        If fewer than `history_length` turns have been played, the remaining
        slots are filled with `None`.

        Returns:
            tuple: State tuple of length `history_length`.
        """
        state = []
        recent_history = self.history[-self.history_length:]

        for guess, black, white in recent_history:
            record = (*guess, black, white)
            state.append(record)

        while len(state) < self.history_length:
            state.append(None)
        return tuple(state)
    
    def reset(self, secret: Optional[Tuple[int, ...]] = None) -> Tuple:
        """
        Reset the environment for a new episode.

        Args:
            secret (tuple | None): Optional fixed secret code for testing.

        Returns:
            tuple: Initial state.
        """
        if secret is not None:
            self.secret = secret
        else:
            self.secret = self.all_codes[np.random.randint(self.num_codes)]
        
        self.turn = 0
        self.history = []
        self.done = False
        
        return self.get_state()
    
    def step(self, action: int) -> Tuple[Tuple, float, bool, dict]:
        """
        Take one environment step.

        Args:
            action (int): Action index representing a guessed code.

        Returns:
            tuple: (next_state, reward, done, info)
        """
        if self.done:
            raise ValueError("Episode done. Call reset().")
        
        guess = self.action_to_code(action)

        # `self.secret` is Optional for type-checking; it should always be set by `reset()`.
        if self.secret is None:
            raise ValueError("Secret code is not set. Call reset() before step().")
        secret = self.secret

        black, white = self.compute_feedback(guess, secret)
        
        self.turn += 1
        self.history.append((guess, black, white))
        
        # Check win/lose
        won = (black == self.num_positions)
        out_of_turns = (self.turn >= self.max_turns)
        self.done = won or out_of_turns
        
        # Reward
        if won:
            reward = 10.0
        else:
            reward = -1.0
        
        info = {
            'guess': guess,
            'black': black,
            'white': white,
            'turn': self.turn,
            'won': won,
        }
        
        return self.get_state(), reward, self.done, info


def compute_consistent_codes(history, all_codes, compute_feedback_fn):
    """Find codes consistent with all feedback received."""
    consistent = []
    for candidate in all_codes:
        ok = True
        for guess, black, white in history:
            b, w = compute_feedback_fn(guess, candidate)
            if b != black or w != white:
                ok = False
                break
        if ok:
            consistent.append(candidate)
    return consistent


def test_baselines():
    """Compare random vs optimal baselines."""
    env = MastermindEnv(num_positions=2, num_colors=4, max_turns=6, history_length=4)
    
    print("=" * 60)
    print("Mastermind")
    print("=" * 60)
    print(f"Positions: {env.num_positions}")
    print(f"Colors: {env.num_colors} (0, 1, 2, 3)")
    print(f"Total codes: {env.num_codes}")
    print(f"Max turns: {env.max_turns}")
    print(f"History length in state: {env.history_length}")
    print()
    
    # Random agent
    print("Random agent (ignores feedback):")
    wins = 0
    for _ in range(1000):
        env.reset()
        while not env.done:
            action = np.random.randint(env.num_codes)
            env.step(action)
        if env.history[-1][1] == env.num_positions:  # black == num_positions
            wins += 1
    print(f"  Win rate: {wins/1000:.1%}")
    print()
    
    # Consistent random agent
    print("Consistent random agent (uses feedback):")
    wins = 0
    total_turns = []
    for _ in range(1000):
        env.reset()
        while not env.done:
            consistent = compute_consistent_codes(
                env.history, env.all_codes, env.compute_feedback
            )
            code = consistent[np.random.randint(len(consistent))]
            action = env.code_to_action(code)
            env.step(action)
        if env.history[-1][1] == env.num_positions:
            wins += 1
            total_turns.append(env.turn)
    print(f"  Win rate: {wins/1000:.1%}")
    print(f"  Avg turns: {np.mean(total_turns):.2f}")


if __name__ == "__main__":
    test_baselines()
