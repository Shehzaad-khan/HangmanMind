#!/usr/bin/env python3
"""
Data cleaning script for Hangman corpus and test files.
Removes multi-word entries, single letters, duplicates, and non-alphabetic words.
"""

def clean_word(word):
    """Clean a single word."""
    return word.strip().lower()

def is_valid_word(word):
    """Check if word is valid for Hangman."""
    return (
        len(word) >= 2 and  # At least 2 characters
        len(word) <= 30 and  # Max 30 characters
        word.isalpha() and  # Only letters
        ' ' not in word  # No spaces
    )

def clean_file(input_path, output_path):
    """Clean a data file."""
    print(f"\nCleaning {input_path}...")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        words = [clean_word(line) for line in f if line.strip()]
    
    # Filter valid words
    valid_words = [w for w in words if is_valid_word(w)]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_words = []
    for word in valid_words:
        if word not in seen:
            seen.add(word)
            unique_words.append(word)
    
    # Write cleaned data
    with open(output_path, 'w', encoding='utf-8') as f:
        for word in unique_words:
            f.write(f"{word}\n")
    
    # Report stats
    removed = len(words) - len(unique_words)
    print(f"✓ Original: {len(words)} words")
    print(f"✓ Cleaned: {len(unique_words)} words")
    print(f"✓ Removed: {removed} words")
    print(f"✓ Saved to: {output_path}")

if __name__ == "__main__":
    print("="*70)
    print("DATA CLEANING FOR HANGMAN")
    print("="*70)
    
    # Clean corpus
    clean_file('Data/corpus.txt', 'Data/corpus_cleaned.txt')
    
    # Clean test set
    clean_file('Data/test.txt', 'Data/test_cleaned.txt')
    
    print("\n" + "="*70)
    print("✅ DATA CLEANING COMPLETE!")
    print("="*70)
