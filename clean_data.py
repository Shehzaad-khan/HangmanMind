#!/usr/bin/env python3
"""
COMPREHENSIVE Data Cleaning Script for Hangman
Handles ALL edge cases: padding, case, repeating letters, special chars, etc.
"""

import re
from collections import Counter

def clean_word(word):
    """Clean a single word - remove padding, lowercase, strip."""
    # Remove leading/trailing whitespace and padding
    word = word.strip()
    
    # Remove common padding characters (spaces, tabs, underscores, hyphens)
    word = word.strip(' \t\n\r_-')
    
    # Convert to lowercase
    word = word.lower()
    
    # Remove any non-alphabetic characters
    word = re.sub(r'[^a-z]', '', word)
    
    return word

def has_excessive_repeating_letters(word, max_consecutive=3):
    """Check if word has excessive repeating letters (e.g., 'zzz', 'qqqqq')."""
    for i in range(len(word) - max_consecutive + 1):
        if len(set(word[i:i+max_consecutive])) == 1:
            return True
    return False

def has_unrealistic_letter_patterns(word):
    """Check for unrealistic letter combinations."""
    # Check for 4+ same letters in a row (like 'eeee', 'zzzz')
    if re.search(r'(.)\1{3,}', word):
        return True
    
    # Check for words with only 1 unique letter repeated (like 'aaa', 'zzzz')
    if len(set(word)) == 1 and len(word) > 2:
        return True
    
    # Check for suspicious patterns (all consonants with no vowels in long words)
    if len(word) >= 6:
        vowels = set('aeiou')
        if not any(c in vowels for c in word):
            # Exception: some valid words like 'rhythms', 'syzygy'
            # Allow if it has 'y'
            if 'y' not in word:
                return True
    
    return False

def has_invalid_suffix_patterns(word):
    """Check for invalid suffix patterns that don't exist in English."""
    # Words ending with unusual patterns
    invalid_endings = [
        'qq', 'qx', 'qz', 'vv', 'jj', 'kk', 'xx', 'zz',
        'bq', 'cq', 'dq', 'fq', 'gq', 'jq', 'kq', 'lq', 'mq', 'nq', 'pq', 'rq', 'sq', 'tq', 'vq', 'wq', 'xq', 'yq', 'zq'
    ]
    
    for ending in invalid_endings:
        if word.endswith(ending):
            return True
    
    return False

def has_invalid_prefix_patterns(word):
    """Check for invalid prefix patterns."""
    # Words starting with unusual patterns
    invalid_starts = [
        'qq', 'qz', 'qx', 'vv', 'xx', 'zz', 'bq', 'cq', 'fq', 'jq', 'kq', 'pq', 'vq', 'xq', 'zq'
    ]
    
    for start in invalid_starts:
        if word.startswith(start):
            return True
    
    return False

def calculate_letter_diversity(word):
    """Calculate letter diversity ratio (unique letters / total letters)."""
    if len(word) == 0:
        return 0
    return len(set(word)) / len(word)

def is_valid_word(word):
    """
    Comprehensive validation for Hangman words.
    
    Checks:
    1. Basic length constraints (2-30 characters)
    2. Only alphabetic characters
    3. No spaces or special characters
    4. No excessive repeating letters (zzz, qqqqq)
    5. No unrealistic letter patterns
    6. No invalid prefix/suffix patterns
    7. Reasonable letter diversity (not just 'aaaaa')
    8. Not a single letter
    9. Not empty
    """
    # Empty check
    if not word:
        return False
    
    # Length constraints
    if len(word) < 2 or len(word) > 30:
        return False
    
    # Only alphabetic
    if not word.isalpha():
        return False
    
    # No spaces (double check)
    if ' ' in word or '\t' in word or '\n' in word:
        return False
    
    # Check for excessive repeating letters
    if has_excessive_repeating_letters(word, max_consecutive=3):
        return False
    
    # Check for unrealistic patterns
    if has_unrealistic_letter_patterns(word):
        return False
    
    # Check for invalid suffix patterns
    if has_invalid_suffix_patterns(word):
        return False
    
    # Check for invalid prefix patterns
    if has_invalid_prefix_patterns(word):
        return False
    
    # Letter diversity check (at least 2 unique letters for words > 2 chars)
    if len(word) > 2:
        diversity = calculate_letter_diversity(word)
        if diversity < 0.3:  # Less than 30% unique letters
            return False
    
    return True

def analyze_cleaning_stats(original_words, cleaned_words):
    """Detailed analysis of what was removed."""
    stats = {
        'too_short': 0,
        'too_long': 0,
        'non_alphabetic': 0,
        'has_spaces': 0,
        'repeating_letters': 0,
        'unrealistic_patterns': 0,
        'invalid_suffix': 0,
        'invalid_prefix': 0,
        'low_diversity': 0,
        'duplicates': 0,
        'empty': 0
    }
    
    cleaned_set = set(cleaned_words)
    seen = set()
    
    for word in original_words:
        cleaned = clean_word(word)
        
        # Empty check
        if not cleaned:
            stats['empty'] += 1
            continue
        
        # Duplicate check
        if cleaned in seen:
            stats['duplicates'] += 1
            continue
        seen.add(cleaned)
        
        # Skip if it passed validation
        if cleaned in cleaned_set:
            continue
        
        # Analyze why it failed
        if len(cleaned) < 2:
            stats['too_short'] += 1
        elif len(cleaned) > 30:
            stats['too_long'] += 1
        elif not cleaned.isalpha():
            stats['non_alphabetic'] += 1
        elif ' ' in cleaned or '\t' in cleaned:
            stats['has_spaces'] += 1
        elif has_excessive_repeating_letters(cleaned, 3):
            stats['repeating_letters'] += 1
        elif has_unrealistic_letter_patterns(cleaned):
            stats['unrealistic_patterns'] += 1
        elif has_invalid_suffix_patterns(cleaned):
            stats['invalid_suffix'] += 1
        elif has_invalid_prefix_patterns(cleaned):
            stats['invalid_prefix'] += 1
        elif len(cleaned) > 2 and calculate_letter_diversity(cleaned) < 0.3:
            stats['low_diversity'] += 1
    
    return stats

def clean_file(input_path, output_path):
    """Clean a data file with comprehensive validation."""
    print(f"\n{'='*70}")
    print(f"Cleaning {input_path}...")
    print('='*70)
    
    # Read all words
    with open(input_path, 'r', encoding='utf-8') as f:
        original_words = [line.strip() for line in f if line.strip()]
    
    print(f"📊 Original words loaded: {len(original_words)}")
    
    # Clean each word
    cleaned_words = []
    for word in original_words:
        cleaned = clean_word(word)
        if cleaned:  # Not empty
            cleaned_words.append(cleaned)
    
    print(f"📊 After basic cleaning: {len(cleaned_words)}")
    
    # Filter valid words
    valid_words = [w for w in cleaned_words if is_valid_word(w)]
    print(f"📊 After validation: {len(valid_words)}")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_words = []
    duplicate_count = 0
    for word in valid_words:
        if word not in seen:
            seen.add(word)
            unique_words.append(word)
        else:
            duplicate_count += 1
    
    print(f"📊 After removing duplicates: {len(unique_words)} (removed {duplicate_count} duplicates)")
    
    # Analyze what was removed
    print(f"\n📋 DETAILED CLEANING REPORT:")
    stats = analyze_cleaning_stats(original_words, unique_words)
    print(f"   ❌ Empty/whitespace only: {stats['empty']}")
    print(f"   ❌ Too short (< 2 chars): {stats['too_short']}")
    print(f"   ❌ Too long (> 30 chars): {stats['too_long']}")
    print(f"   ❌ Non-alphabetic chars: {stats['non_alphabetic']}")
    print(f"   ❌ Has spaces/tabs: {stats['has_spaces']}")
    print(f"   ❌ Excessive repeating (zzz, qqqqq): {stats['repeating_letters']}")
    print(f"   ❌ Unrealistic patterns (eeee, all consonants): {stats['unrealistic_patterns']}")
    print(f"   ❌ Invalid suffix (qq, zz endings): {stats['invalid_suffix']}")
    print(f"   ❌ Invalid prefix (qq, zz starts): {stats['invalid_prefix']}")
    print(f"   ❌ Low letter diversity (<30%): {stats['low_diversity']}")
    print(f"   ❌ Duplicates: {stats['duplicates']}")
    
    total_removed = len(original_words) - len(unique_words)
    print(f"\n✅ FINAL: {len(unique_words)} clean words (removed {total_removed})")
    
    # Show sample of removed words
    removed_words = [clean_word(w) for w in original_words if clean_word(w) not in seen and clean_word(w)]
    if removed_words:
        print(f"\n🔍 Sample of removed words (first 20):")
        for word in removed_words[:20]:
            print(f"   - '{word}'")
    
    # Write cleaned data
    with open(output_path, 'w', encoding='utf-8') as f:
        for word in unique_words:
            f.write(f"{word}\n")
    
    print(f"\n💾 Saved to: {output_path}")
    
    # Show letter distribution in cleaned data
    all_letters = ''.join(unique_words)
    letter_freq = Counter(all_letters)
    print(f"\n📈 Letter frequency in cleaned data (top 10):")
    for letter, count in letter_freq.most_common(10):
        pct = (count / len(all_letters)) * 100
        print(f"   {letter}: {count:,} ({pct:.2f}%)")

if __name__ == "__main__":
    print("="*70)
    print("COMPREHENSIVE DATA CLEANING FOR HANGMAN")
    print("Handles: padding, case, repeating letters, special chars, etc.")
    print("="*70)
    
    # Clean corpus
    clean_file('Data/corpus.txt', 'Data/corpus_cleaned.txt')
    
    # Clean test set
    clean_file('Data/test.txt', 'Data/test_cleaned.txt')
    
    print("\n" + "="*70)
    print("✅ DATA CLEANING COMPLETE!")
    print("="*70)
