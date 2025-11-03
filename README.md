# Hangman AI Solver - HMM + Deep Reinforcement Learning

## Team Members

| Name | SRN |
|------|-----|
| Mohammed Musharraf | PES2UG23CS915 |
| Mohammed Shehzaad Khan | PES2U23CS349 |
| Mohammed Bilal | PES2UG23CS344 |
| Mohammed Aahil | PES2UG23CS342 |

---

## 🎯 Final Results

- **Success Rate:** 94.40% on 2000 test games
- **Average Wrong Guesses:** 2.13 per game
- **Repeated Guesses:** 0
- **Training Win Rate (Pure RL):** 7.18%
- **Testing Win Rate (Hybrid System):** 94.40%

## 📁 Project Structure

```
ML-Hackathon/
├── Data/
│   ├── corpus.txt           # 50,000 training words
│   └── test.txt             # 2,000 test words
├── Untitled14.ipynb         # Main implementation notebook
├── Analysis_Report.md       # Detailed analysis report
└── README.md                # This file
```

## 🏗️ Architecture

### Three-Part System

**1. Hidden Markov Model (HMM)**
- Position-based states (0, 1, 2, ..., word_length-1)
- Letter emissions (A-Z)
- 24 separate HMMs for word lengths 1-24
- Laplace smoothing (α=1.0) for unseen patterns
- Captures positional letter patterns in English

**2. Word Filtering System**
- Matches current pattern with corpus words
- When ≤20 words match, uses direct letter frequency
- Most powerful component of the system
- Provides accurate predictions for narrow search spaces

**3. Deep Q-Network (DQN) Agent**
- State representation: 619 dimensions
  - Masked word (540 dims): 20 positions × 27 one-hot features
  - Guessed letters (26 dims): Binary vector
  - Lives remaining (1 dim): Normalized
  - HMM probabilities (26 dims)
  - Word filter probabilities (26 dims)
- Network architecture: 619 → 256 → 128 → 64 → 26
- Experience replay buffer: 10,000 transitions
- Target network updated every 10 episodes
- Epsilon-greedy exploration: 1.0 → 0.01

**Hybrid Strategy**
- If matching words ≤ 20: Use word filter directly
- Otherwise: Blend all sources
  - 50% word filtering
  - 30% HMM predictions
  - 20% DQN Q-values

## 🚀 Quick Start

### Setup

1. **Install dependencies:**
```bash
pip install torch tqdm matplotlib numpy pickle
```

2. **Prepare data files:**
- Place `corpus.txt` in root directory (50,000 words)
- Place `test_words.txt` in root directory (2,000 words)

### Running the Notebook

The notebook `Untitled14.ipynb` contains three main parts:

**Part 1: HMM Training**
- Loads and preprocesses corpus
- Trains 24 HMMs (one per word length)
- Creates word matcher for filtering
- Saves models: `hangman_models.pkl`, `word_matcher.pkl`
- Training time: ~1 minute

**Part 2: RL Agent Training**
- Loads trained HMMs
- Creates Hangman environment
- Trains DQN agent for 5000 episodes
- Saves model: `dqn_agent.pth`
- Training time: ~19 minutes on GPU

**Part 3: Evaluation**
- Loads all models
- Creates hybrid agent
- Evaluates on 2000 test words
- Generates visualizations
- Saves results: `evaluation_results.pkl`
- Evaluation time: ~2 minutes

### Expected Output

```
HANGMAN HMM TRAINING
Loaded 50000 words
Training HMMs for each word length...
✅ HMM Training Complete!

HANGMAN RL TRAINING
Training: 100%|██████████| 5000/5000
Final Win Rate: 7.18%
✅ Training Complete!

HANGMAN EVALUATION
Evaluating Hybrid Agent: 100%|██████████| 2000/2000
Success Rate: 94.40%
✅ Evaluation Complete!
```

## 📊 Key Findings

### What Worked

✅ **Hybrid approach dominated** - Combining word filtering, HMM, and RL achieved 94.40% vs 7.18% for pure RL

✅ **Word filtering was critical** - Direct corpus matching provided the strongest signal

✅ **HMM captured language patterns** - Position-specific letter probabilities worked well

### What Didn't Work

❌ **Pure RL struggled** - 619-dimensional state space made learning difficult

❌ **Sparse rewards** - Mostly negative feedback hindered learning

❌ **Too conservative exploration** - 700 episodes to reach minimum epsilon was too slow

## 🎯 Reward Function

- **Correct guess:** +10 per position revealed + 100 if won
- **Wrong guess:** -15 (+ -100 if lost)
- **Repeated guess:** -20 (heavy efficiency penalty)

## � Performance by Word Length

- **Short words (2-4 letters):** ~50-80% win rate (limited context)
- **Medium words (5-9 letters):** ~95%+ win rate (optimal)
- **Long words (10-15 letters):** ~95%+ win rate (many clues)
- **Very long words (16+ letters):** ~100% win rate (extensive context)

## 🔧 Training Parameters

| Parameter | Value |
|-----------|-------|
| Episodes | 5000 |
| Batch Size | 64 |
| Learning Rate | 0.001 |
| Gamma (Discount) | 0.95 |
| Epsilon Start | 1.0 |
| Epsilon Min | 0.01 |
| Epsilon Decay | 0.995 |
| Replay Buffer | 10,000 |
| Target Update | Every 10 episodes |

## 📝 Files Generated

- `hangman_models.pkl` - Trained HMM models for all word lengths
- `word_matcher.pkl` - Word filtering system
- `dqn_agent.pth` - Trained DQN agent weights
- `training_results.png` - Training curves visualization
- `evaluation_results.pkl` - Evaluation statistics
- `evaluation_results.png` - Evaluation visualizations

## � Key Lessons

1. **Domain knowledge beats pure learning** - Explicit word matching outperformed neural networks
2. **System design matters** - Intelligent combination of techniques is more powerful than any single method
3. **RL as refinement** - Use RL to handle edge cases, not learn entire strategy from scratch
4. **Simple can be better** - Word filtering was simpler and more effective than complex neural networks

## � References

- Hidden Markov Models for sequence prediction
- Deep Q-Networks (DQN) for reinforcement learning
- Experience replay and target networks for stable learning
- Epsilon-greedy exploration strategy

---

**PES University - Machine Learning Hackathon 2025**  
**Date:** November 3, 2025
