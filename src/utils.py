"""
Utility functions for state encoding and reward calculation.
"""

import numpy as np
from typing import List, Set, Optional, Dict


def encode_state(masked_word: List[Optional[str]], 
                 guessed_letters: Set[str], 
                 hmm_probs: Dict[str, float],
                 lives: int,
                 word_length: int,
                 max_word_length: int = 30) -> np.ndarray:
    """
    Encode game state into a feature vector for RL agent.
    
    Args:
        masked_word: List of characters or None for blanks
        guessed_letters: Set of guessed letters
        hmm_probs: Dictionary mapping letters to probabilities from HMM
        lives: Remaining lives
        word_length: Length of the word
        max_word_length: Maximum word length (for padding)
    
    Returns:
        Feature vector as numpy array
    """
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    
    # 1. Masked word encoding (one-hot per position, max_word_length positions)
    # Each position gets 27-dim vector (26 letters + 1 for blank)
    masked_encoding = np.zeros((max_word_length, 27))
    for i in range(min(len(masked_word), max_word_length)):
        if masked_word[i] is None:
            masked_encoding[i, 26] = 1.0  # Blank
        else:
            char_idx = ord(masked_word[i]) - ord('a')
            if 0 <= char_idx < 26:
                masked_encoding[i, char_idx] = 1.0
    masked_encoding = masked_encoding.flatten()
    
    # 2. Guessed letters binary vector (26-dim)
    guessed_encoding = np.zeros(26)
    for char in guessed_letters:
        char_idx = ord(char.lower()) - ord('a')
        if 0 <= char_idx < 26:
            guessed_encoding[char_idx] = 1.0
    
    # 3. HMM probability vector (26-dim)
    hmm_encoding = np.zeros(26)
    for char, prob in hmm_probs.items():
        char_idx = ord(char.lower()) - ord('a')
        if 0 <= char_idx < 26:
            hmm_encoding[char_idx] = prob
    
    # 4. Lives (normalized)
    lives_normalized = lives / 6.0  # Assuming max_lives = 6
    
    # 5. Word length (normalized)
    length_normalized = word_length / max_word_length
    
    # 6. Number of revealed letters (normalized)
    revealed_count = sum(1 for char in masked_word if char is not None)
    revealed_normalized = revealed_count / max_word_length if max_word_length > 0 else 0.0
    
    # Combine all features
    state_vector = np.concatenate([
        masked_encoding,
        guessed_encoding,
        hmm_encoding,
        [lives_normalized, length_normalized, revealed_normalized]
    ])
    
    return state_vector


def get_reward(state: Dict[str, any], 
               info: Dict[str, any],
               prev_state: Optional[Dict[str, any]] = None) -> float:
    """
    Calculate reward based on game state and action result.
    
    This is a more sophisticated reward function that considers:
    - Correct guesses: positive reward
    - Wrong guesses: negative reward (penalties increase with lives lost)
    - Repeated guesses: negative reward
    - Winning: large positive reward
    - Losing: negative reward
    
    Args:
        state: Current game state
        info: Info from environment step
        prev_state: Previous state (optional)
    
    Returns:
        Reward value
    """
    reward = 0.0
    
    # Check if action was correct
    if info.get('correct', False):
        # Correct guess
        positions_revealed = info.get('positions_revealed', [])
        reward += 1.0 + 0.5 * len(positions_revealed)
        
        # Bonus for winning
        if info.get('won', False):
            reward += 10.0
    
    elif info.get('repeated', False):
        # Repeated guess penalty
        reward -= 2.0
    
    else:
        # Wrong guess
        reward -= 1.0
        
        # Additional penalty if losing
        lives_remaining = info.get('lives_remaining', 0)
        if lives_remaining == 0:
            reward -= 5.0
    
    return reward


def calculate_final_score(success_rate: float,
                          total_wrong_guesses: int,
                          total_repeated_guesses: int,
                          num_games: int = 2000) -> float:
    """
    Calculate final score according to the competition formula.
    
    Formula: (Success Rate * 2000) - (Total Wrong Guesses * 5) - (Total Repeated Guesses * 2)
    
    Args:
        success_rate: Win rate (0.0 to 1.0)
        total_wrong_guesses: Total number of wrong guesses across all games
        total_repeated_guesses: Total number of repeated guesses across all games
        num_games: Total number of games played
    
    Returns:
        Final score
    """
    score = (success_rate * num_games) - (total_wrong_guesses * 5) - (total_repeated_guesses * 2)
    return score

