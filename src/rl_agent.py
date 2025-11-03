"""
Reinforcement Learning Agent for Hangman game.

This module implements a Q-learning agent that uses HMM probabilities
to make intelligent letter guesses.
"""

import numpy as np
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import pickle


class QLearningAgent:
    """
    Q-learning agent for Hangman game.
    
    The agent uses a state-action value function (Q-table) to learn
    the optimal policy for guessing letters in Hangman.
    """
    
    def __init__(self,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 max_word_length: int = 30):
        """
        Initialize Q-learning agent.
        
        Args:
            learning_rate: Learning rate (alpha) for Q-value updates
            discount_factor: Discount factor (gamma) for future rewards
            epsilon: Initial exploration rate
            epsilon_decay: Rate at which epsilon decays
            epsilon_min: Minimum epsilon value
            max_word_length: Maximum word length to handle
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.max_word_length = max_word_length
        
        # Q-table: state -> action -> Q-value
        # Since state space is large, we use a function approximation approach
        # We'll use a simplified state representation key
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Track statistics
        self.training_stats = {
            'episodes': 0,
            'total_reward': 0.0,
            'wins': 0,
            'losses': 0,
            'wrong_guesses': 0,
            'repeated_guesses': 0
        }
        
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.alphabet_size = len(self.alphabet)
    
    def get_state_key(self, state_features: np.ndarray, hmm_probs: Dict[str, float]) -> str:
        """
        Convert state features to a hashable key for Q-table lookup.
        
        Since the full state space is too large, we use a simplified representation:
        - Word length
        - Number of revealed letters
        - Lives remaining
        - Top 3 HMM probabilities (letters)
        
        Args:
            state_features: State feature vector
            hmm_probs: HMM probability distribution
        
        Returns:
            State key as string
        """
        # Extract key features
        word_length = int(state_features[-2] * self.max_word_length)
        revealed_count = int(state_features[-1] * self.max_word_length)
        lives = int(state_features[-3] * 6)  # Assuming max_lives = 6
        
        # Get top 3 HMM predictions
        sorted_probs = sorted(hmm_probs.items(), key=lambda x: x[1], reverse=True)
        top_letters = ''.join([char for char, _ in sorted_probs[:3]])
        
        # Create state key
        state_key = f"len_{word_length}_rev_{revealed_count}_lives_{lives}_hmm_{top_letters}"
        return state_key
    
    def get_available_actions(self, guessed_letters: set) -> List[str]:
        """
        Get list of available actions (unguessed letters).
        
        Args:
            guessed_letters: Set of already guessed letters
        
        Returns:
            List of available letter actions
        """
        return [char for char in self.alphabet if char not in guessed_letters]
    
    def select_action(self,
                     state_features: np.ndarray,
                     hmm_probs: Dict[str, float],
                     guessed_letters: set,
                     use_hmm_prior: bool = True) -> str:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state_features: State feature vector
            hmm_probs: HMM probability distribution
            guessed_letters: Set of guessed letters
            use_hmm_prior: Whether to use HMM probabilities as prior
        
        Returns:
            Selected letter action
        """
        available_actions = self.get_available_actions(guessed_letters)
        
        if not available_actions:
            # Should not happen in normal play
            return random.choice(self.alphabet)
        
        # Exploration: random action
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        
        # Exploitation: choose best action based on Q-values
        state_key = self.get_state_key(state_features, hmm_probs)
        
        # Get Q-values for available actions
        action_values = {}
        for action in available_actions:
            q_value = self.q_table[state_key][action]
            
            # Combine Q-value with HMM probability as prior
            if use_hmm_prior:
                hmm_prior = hmm_probs.get(action, 0.0)
                # Weighted combination: Q-value + HMM prior
                # Using 1.5 weight gives more trust to HMM (based on empirical testing)
                action_values[action] = q_value + 1.5 * hmm_prior
            else:
                action_values[action] = q_value
        
        # Choose action with highest value
        best_action = max(action_values.items(), key=lambda x: x[1])[0]
        return best_action
    
    def update_q_value(self,
                       state_features: np.ndarray,
                       action: str,
                       reward: float,
                       next_state_features: Optional[np.ndarray],
                       next_hmm_probs: Optional[Dict[str, float]],
                       next_guessed_letters: Optional[set],
                       done: bool,
                       current_hmm_probs: Optional[Dict[str, float]] = None):
        """
        Update Q-value using Q-learning update rule.
        
        Q(s, a) = Q(s, a) + alpha * [r + gamma * max Q(s', a') - Q(s, a)]
        
        Args:
            state_features: Current state features
            action: Action taken
            reward: Reward received
            next_state_features: Next state features (None if done)
            next_hmm_probs: Next HMM probabilities (None if done)
            next_guessed_letters: Next set of guessed letters (None if done)
            done: Whether episode is finished
            current_hmm_probs: Current HMM probabilities (optional, falls back to uniform)
        """
        # Use provided hmm_probs or fallback to uniform
        if current_hmm_probs is None:
            current_hmm_probs = self._get_current_hmm_probs(state_features)
        
        state_key = self.get_state_key(state_features, current_hmm_probs)
        
        current_q = self.q_table[state_key][action]
        
        if done:
            # Terminal state: Q(s, a) = Q(s, a) + alpha * [r - Q(s, a)]
            target_q = reward
        else:
            # Non-terminal: Q(s, a) = Q(s, a) + alpha * [r + gamma * max Q(s', a') - Q(s, a)]
            if next_hmm_probs is None:
                next_hmm_probs = self._get_current_hmm_probs(next_state_features)
            next_state_key = self.get_state_key(next_state_features, next_hmm_probs)
            next_available = self.get_available_actions(next_guessed_letters)
            
            if next_available:
                next_max_q = max([self.q_table[next_state_key][a] for a in next_available])
                target_q = reward + self.discount_factor * next_max_q
            else:
                target_q = reward
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (target_q - current_q)
        self.q_table[state_key][action] = new_q
    
    def _get_current_hmm_probs(self, state_features: np.ndarray) -> Dict[str, float]:
        """Get HMM probabilities from state features (placeholder)."""
        # HMM probs should be passed separately, but this is a fallback
        # Return uniform distribution
        return {char: 1.0 / 26 for char in self.alphabet}
    
    def decay_epsilon(self):
        """Decay epsilon value for exploration."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_epsilon(self) -> float:
        """Get current epsilon value."""
        return self.epsilon
    
    def update_stats(self, reward: float, won: bool, wrong_count: int, repeated_count: int):
        """Update training statistics."""
        self.training_stats['episodes'] += 1
        self.training_stats['total_reward'] += reward
        if won:
            self.training_stats['wins'] += 1
        else:
            self.training_stats['losses'] += 1
        self.training_stats['wrong_guesses'] += wrong_count
        self.training_stats['repeated_guesses'] += repeated_count
    
    def get_stats(self) -> Dict:
        """Get training statistics."""
        stats = self.training_stats.copy()
        if stats['episodes'] > 0:
            stats['win_rate'] = stats['wins'] / stats['episodes']
            stats['avg_reward'] = stats['total_reward'] / stats['episodes']
        else:
            stats['win_rate'] = 0.0
            stats['avg_reward'] = 0.0
        return stats
    
    def save(self, filepath: str):
        """Save agent to file."""
        agent_data = {
            'q_table': {k: dict(v) for k, v in self.q_table.items()},
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'max_word_length': self.max_word_length,
            'training_stats': self.training_stats
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(agent_data, f)
        
        print(f"Agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load agent from file."""
        with open(filepath, 'rb') as f:
            agent_data = pickle.load(f)
        
        self.q_table = defaultdict(lambda: defaultdict(float))
        for k, v in agent_data['q_table'].items():
            self.q_table[k] = defaultdict(float, v)
        
        self.learning_rate = agent_data['learning_rate']
        self.discount_factor = agent_data['discount_factor']
        self.epsilon = agent_data['epsilon']
        self.epsilon_decay = agent_data['epsilon_decay']
        self.epsilon_min = agent_data['epsilon_min']
        self.max_word_length = agent_data['max_word_length']
        self.training_stats = agent_data['training_stats']
        
        print(f"Agent loaded from {filepath}")


class HMGPriorAgent:
    """
    Simple agent that uses only HMM probabilities (baseline).
    This can be used for comparison with the RL agent.
    """
    
    def __init__(self):
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
    
    def select_action(self, hmm_probs: Dict[str, float], guessed_letters: set) -> str:
        """
        Select action based solely on HMM probabilities.
        
        Args:
            hmm_probs: HMM probability distribution
            guessed_letters: Set of guessed letters
        
        Returns:
            Selected letter action
        """
        available = [char for char in self.alphabet if char not in guessed_letters]
        
        if not available:
            return random.choice(self.alphabet)
        
        # Choose letter with highest HMM probability
        available_probs = {char: hmm_probs.get(char, 0.0) for char in available}
        best_letter = max(available_probs.items(), key=lambda x: x[1])[0]
        return best_letter

