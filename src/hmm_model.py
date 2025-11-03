"""
Hidden Markov Model for Hangman letter prediction.

The HMM models character sequences in words. We use position-aware character transitions
to predict which letters are likely to appear in each position of a word.
"""

import numpy as np
from collections import defaultdict, Counter
import pickle


class HangmanHMM:
    """
    Hidden Markov Model for Hangman game.
    
    Architecture:
    - Hidden states: Character positions in the word (0 to max_length-1)
    - Emissions: Characters observed at each position
    - Transitions: Character-to-character transitions within words
    - We also track character frequency at each position
    
    For inference, we use:
    1. Character bigram/trigram probabilities
    2. Position-specific character frequencies
    3. Global character frequencies
    """
    
    def __init__(self, max_word_length=20):
        """
        Initialize HMM model.
        
        Args:
            max_word_length: Maximum word length to model
        """
        self.max_word_length = max_word_length
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.char_to_idx = {char: idx for idx, char in enumerate(self.alphabet)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        
        # Character frequencies at each position
        self.position_char_counts = defaultdict(lambda: Counter())  # position -> char -> count
        
        # Character bigram probabilities (char1 -> char2)
        self.bigram_counts = defaultdict(lambda: Counter())  # char1 -> char2 -> count
        
        # Character trigram probabilities
        self.trigram_counts = defaultdict(lambda: defaultdict(lambda: Counter()))  # char1, char2 -> char3 -> count
        
        # Global character frequencies
        self.global_char_counts = Counter()
        
        # Word length distribution
        self.length_counts = Counter()
        
        # Total words trained on
        self.total_words = 0
        
    def train(self, words):
        """
        Train HMM on a list of words.
        
        Args:
            words: List of normalized words (lowercase, alphabetic only)
        """
        print(f"Training HMM on {len(words)} words...")
        
        for word in words:
            if len(word) == 0:
                continue
                
            word_len = len(word)
            self.length_counts[word_len] += 1
            self.total_words += 1
            
            # Update global character counts
            for char in word:
                self.global_char_counts[char] += 1
            
            # Update position-specific character counts
            for pos, char in enumerate(word):
                if pos < self.max_word_length:
                    self.position_char_counts[pos][char] += 1
            
            # Update bigram counts
            for i in range(len(word) - 1):
                char1 = word[i]
                char2 = word[i + 1]
                self.bigram_counts[char1][char2] += 1
            
            # Update trigram counts
            for i in range(len(word) - 2):
                char1 = word[i]
                char2 = word[i + 1]
                char3 = word[i + 2]
                self.trigram_counts[char1][char2][char3] += 1
        
        print(f"Training complete. Processed {self.total_words} words.")
        print(f"Word length range: {min(self.length_counts.keys())} - {max(self.length_counts.keys())}")
    
    def get_position_probabilities(self, position, word_length, guessed_letters=set()):
        """
        Get probability distribution over characters for a specific position.
        
        Args:
            position: Position in the word (0-indexed)
            word_length: Length of the word
            guessed_letters: Set of already guessed letters (to exclude)
        
        Returns:
            Dictionary mapping characters to probabilities
        """
        probs = {}
        total_count = sum(self.position_char_counts[position].values())
        
        # Smoothing parameter
        smoothing = 0.1
        
        if total_count > 0:
            for char in self.alphabet:
                if char in guessed_letters:
                    probs[char] = 0.0
                else:
                    count = self.position_char_counts[position].get(char, 0)
                    probs[char] = (count + smoothing) / (total_count + smoothing * 26)
        else:
            # If no data for this position, use uniform distribution
            unguessed_count = 26 - len(guessed_letters)
            for char in self.alphabet:
                if char in guessed_letters:
                    probs[char] = 0.0
                else:
                    probs[char] = 1.0 / unguessed_count if unguessed_count > 0 else 0.0
        
        return probs
    
    def predict_letter_probabilities(self, masked_word, guessed_letters=set(), word_length=None):
        """
        Predict probability distribution over letters given current game state.
        
        Args:
            masked_word: Current state of the word (e.g., "_ E _ _ _")
                        Can be string with underscores or list of chars/None
            guessed_letters: Set of already guessed letters
            word_length: Length of the word (if None, inferred from masked_word)
        
        Returns:
            Dictionary mapping characters to probabilities
        """
        # Convert masked_word to list format if it's a string
        if isinstance(masked_word, str):
            word_state = [char if char != '_' else None for char in masked_word.replace(' ', '')]
            if word_length is None:
                word_length = len(word_state)
        else:
            word_state = masked_word
            if word_length is None:
                word_length = len(word_state)
        
        # Initialize probability vector
        letter_probs = {char: 0.0 for char in self.alphabet}
        
        # Get revealed positions
        revealed_positions = [i for i, char in enumerate(word_state) if char is not None]
        
        # Combine multiple sources of information:
        
        # 1. Position-specific probabilities for blank positions
        for pos in range(word_length):
            if word_state[pos] is None:  # Blank position
                pos_probs = self.get_position_probabilities(pos, word_length, guessed_letters)
                for char in self.alphabet:
                    letter_probs[char] += pos_probs[char]
        
        # 2. Context from bigrams/trigrams around revealed letters
        for pos in revealed_positions:
            char = word_state[pos]
            if char in self.bigram_counts:
                # Look at characters following this one
                if pos + 1 < word_length and word_state[pos + 1] is None:
                    for next_char, count in self.bigram_counts[char].items():
                        if next_char not in guessed_letters:
                            total = sum(self.bigram_counts[char].values())
                            if total > 0:
                                letter_probs[next_char] += count / total * 2.0  # Weight factor
            
            # Look at characters preceding this one
            if pos > 0 and word_state[pos - 1] is None:
                for prev_char in self.alphabet:
                    if prev_char in self.bigram_counts and char in self.bigram_counts[prev_char]:
                        count = self.bigram_counts[prev_char][char]
                        total = sum(self.bigram_counts[prev_char].values())
                        if total > 0:
                            letter_probs[prev_char] += count / total * 2.0
        
        # 3. Global character frequency (fallback)
        total_global = sum(self.global_char_counts.values())
        if total_global > 0:
            global_weight = 0.5
            for char in self.alphabet:
                if char not in guessed_letters:
                    freq = self.global_char_counts.get(char, 0)
                    letter_probs[char] += (freq / total_global) * global_weight
        
        # Normalize probabilities
        total_prob = sum(letter_probs.values())
        if total_prob > 0:
            letter_probs = {char: prob / total_prob for char, prob in letter_probs.items()}
        else:
            # Uniform distribution over unguessed letters
            unguessed = [c for c in self.alphabet if c not in guessed_letters]
            if len(unguessed) > 0:
                uniform_prob = 1.0 / len(unguessed)
                letter_probs = {char: uniform_prob if char in unguessed else 0.0 for char in self.alphabet}
            else:
                letter_probs = {char: 0.0 for char in self.alphabet}
        
        # Ensure guessed letters have zero probability
        for char in guessed_letters:
            letter_probs[char] = 0.0
        
        return letter_probs
    
    def get_best_letter(self, masked_word, guessed_letters=set(), word_length=None):
        """
        Get the most likely letter to guess next.
        
        Args:
            masked_word: Current state of the word
            guessed_letters: Set of already guessed letters
            word_length: Length of the word
        
        Returns:
            Tuple of (best_letter, probability)
        """
        probs = self.predict_letter_probabilities(masked_word, guessed_letters, word_length)
        if not probs or all(p == 0.0 for p in probs.values()):
            # Fallback: return any unguessed letter
            unguessed = [c for c in self.alphabet if c not in guessed_letters]
            if unguessed:
                return unguessed[0], 1.0 / len(unguessed)
            return None, 0.0
        
        best_letter = max(probs.items(), key=lambda x: x[1])[0]
        return best_letter, probs[best_letter]
    
    def save(self, filepath):
        """Save model to file."""
        model_data = {
            'max_word_length': self.max_word_length,
            'position_char_counts': dict(self.position_char_counts),
            'bigram_counts': {k: dict(v) for k, v in self.bigram_counts.items()},
            'trigram_counts': {k1: {k2: dict(v) for k2, v in v1.items()} 
                              for k1, v1 in self.trigram_counts.items()},
            'global_char_counts': dict(self.global_char_counts),
            'length_counts': dict(self.length_counts),
            'total_words': self.total_words
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.max_word_length = model_data['max_word_length']
        self.position_char_counts = defaultdict(Counter)
        for pos, counts in model_data['position_char_counts'].items():
            self.position_char_counts[pos] = Counter(counts)
        
        self.bigram_counts = defaultdict(Counter)
        for char1, counts in model_data['bigram_counts'].items():
            self.bigram_counts[char1] = Counter(counts)
        
        self.trigram_counts = defaultdict(lambda: defaultdict(Counter))
        for char1, v1 in model_data['trigram_counts'].items():
            for char2, counts in v1.items():
                self.trigram_counts[char1][char2] = Counter(counts)
        
        self.global_char_counts = Counter(model_data['global_char_counts'])
        self.length_counts = Counter(model_data['length_counts'])
        self.total_words = model_data['total_words']
        
        print(f"Model loaded from {filepath}")

