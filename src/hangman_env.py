"""
Hangman Game Environment for Reinforcement Learning.

This module implements a Hangman game environment that follows the OpenAI Gym interface.
"""

import numpy as np
import random
from typing import List, Tuple, Set, Optional, Dict, Any


class HangmanEnv:
    """
    Hangman game environment.
    
    The environment maintains:
    - Current word to guess
    - Masked word state (showing revealed letters)
    - Set of guessed letters
    - Number of remaining lives (wrong guesses allowed)
    - Game over status
    """
    
    def __init__(self, word: str, max_lives: int = 6):
        """
        Initialize Hangman environment.
        
        Args:
            word: The word to guess (must be lowercase)
            max_lives: Maximum number of wrong guesses allowed
        """
        self.word = word.lower()
        self.max_lives = max_lives
        self.lives = max_lives
        self.guessed_letters: Set[str] = set()
        self.masked_word = ['_'] * len(self.word)
        self.done = False
        self.won = False
        self._last_guess = None
        self._last_reward = 0.0
        self._guessed_count = 0  # Total guesses made
        self._wrong_count = 0  # Wrong guesses made
        self._repeated_count = 0  # Repeated guesses
        
    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state.
        
        Returns:
            Initial state observation
        """
        self.lives = self.max_lives
        self.guessed_letters = set()
        self.masked_word = ['_'] * len(self.word)
        self.done = False
        self.won = False
        self._last_guess = None
        self._last_reward = 0.0
        self._guessed_count = 0
        self._wrong_count = 0
        self._repeated_count = 0
        return self._get_state()
    
    def step(self, action: str) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Letter to guess (single character, lowercase)
        
        Returns:
            Tuple of (observation, reward, done, info)
            - observation: Current state
            - reward: Reward for this step
            - done: Whether episode is finished
            - info: Additional information
        """
        if self.done:
            return self._get_state(), 0.0, True, {'message': 'Game already finished'}
        
        action = action.lower()
        
        # Check if letter was already guessed
        if action in self.guessed_letters:
            self._repeated_count += 1
            reward = -2.0  # Penalty for repeated guess
            self._last_guess = action
            self._last_reward = reward
            info = {
                'repeated': True,
                'correct': False,
                'lives_remaining': self.lives
            }
            return self._get_state(), reward, False, info
        
        # Add to guessed letters
        self.guessed_letters.add(action)
        self._guessed_count += 1
        self._last_guess = action
        
        # Check if letter is in the word
        if action in self.word:
            # Correct guess - reveal all instances
            positions_found = []
            for i, char in enumerate(self.word):
                if char == action:
                    self.masked_word[i] = action
                    positions_found.append(i)
            
            # Calculate reward
            num_revealed = len(positions_found)
            reward = 1.0 + 0.5 * num_revealed  # Base reward + bonus for revealing letters
            
            # Check if word is complete
            if '_' not in self.masked_word:
                # Win!
                self.won = True
                self.done = True
                reward += 10.0  # Large bonus for winning
                info = {
                    'won': True,
                    'correct': True,
                    'positions_revealed': positions_found,
                    'lives_remaining': self.lives
                }
            else:
                info = {
                    'won': False,
                    'correct': True,
                    'positions_revealed': positions_found,
                    'lives_remaining': self.lives
                }
        
        else:
            # Wrong guess
            self.lives -= 1
            self._wrong_count += 1
            reward = -1.0  # Penalty for wrong guess
            
            if self.lives <= 0:
                # Lost
                self.done = True
                reward -= 5.0  # Additional penalty for losing
                info = {
                    'won': False,
                    'correct': False,
                    'lives_remaining': 0
                }
            else:
                info = {
                    'won': False,
                    'correct': False,
                    'lives_remaining': self.lives
                }
        
        self._last_reward = reward
        return self._get_state(), reward, self.done, info
    
    def _get_state(self) -> np.ndarray:
        """
        Get current state representation.
        
        Returns:
            State as numpy array (for now, just return a placeholder)
            The actual state encoding will be done in the RL agent.
        """
        # Return a simple state representation
        # The RL agent will encode this with HMM probabilities
        return np.array([len(self.masked_word), self.lives, len(self.guessed_letters)])
    
    def get_masked_word(self) -> str:
        """Get current masked word as string."""
        return ''.join(self.masked_word)
    
    def get_masked_word_list(self) -> List[Optional[str]]:
        """
        Get current masked word as list (None for blank, char for revealed).
        
        Returns:
            List where each element is None (blank) or a character (revealed)
        """
        return [char if char != '_' else None for char in self.masked_word]
    
    def render(self) -> None:
        """Print current game state."""
        masked_str = ' '.join(self.masked_word)
        print(f"Word: {masked_str}")
        print(f"Guessed letters: {sorted(self.guessed_letters)}")
        print(f"Lives remaining: {self.lives}/{self.max_lives}")
        if self.done:
            if self.won:
                print("Status: WON!")
            else:
                print(f"Status: LOST! The word was: {self.word}")
        print("-" * 40)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get game statistics."""
        return {
            'guessed_count': self._guessed_count,
            'wrong_count': self._wrong_count,
            'repeated_count': self._repeated_count,
            'won': self.won,
            'lives_remaining': self.lives
        }


def create_hangman_env(word: str, max_lives: int = 6) -> HangmanEnv:
    """Factory function to create a Hangman environment."""
    return HangmanEnv(word, max_lives)

