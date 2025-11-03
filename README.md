# 🎮 Hangman AI Solver

<div align="center">

### Combining Hidden Markov Models + Deep Reinforcement Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Success Rate](https://img.shields.io/badge/Success%20Rate-94.40%25-brightgreen.svg)](.)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](.)

A sophisticated AI agent that plays Hangman with a 94.40% success rate by combining Hidden Markov Models, Deep Reinforcement Learning, and intelligent word filtering.

</div>

---

## 👥 Team Members

| Name | SRN |
|:----:|:---:|
| **Mohammed Musharraf** | PES2UG23CS915 |
| **Mohammed Shehzaad Khan** | PES2UG23CS349 |
| **Mohammed Bilal** | PES2UG23CS344 |
| **Mohammed Aahil** | PES2UG23CS342 |

---

## 🎯 Results

<div align="center">

| Metric | Value |
|:------:|:-----:|
| 🏆 **Success Rate** | **94.40%** (2000 test games) |
| 📉 **Avg Wrong Guesses** | **2.13** per game |
| 🎲 **Repeated Guesses** | **0** |
| 🔄 **Training Win Rate** (Pure RL) | 7.18% |
| ✨ **Testing Win Rate** (Hybrid) | **94.40%** |

</div>

---

## 🏗️ System Architecture

### Three-Part Hybrid System

Our system uses a **three-part hybrid approach**:

### 1. 🧠 Hidden Markov Model (HMM)

- **States:** Position-based (0, 1, 2, ..., word_length-1)
- **Emissions:** Letters (A-Z)
- **Models:** 24 separate HMMs for word lengths 1-24
- **Smoothing:** Laplace smoothing (α=1.0) for unseen patterns
- **Purpose:** Captures positional letter patterns in English words

### 2. 🔍 Word Filtering System

- Matches current pattern against corpus words
- When ≤20 words match, uses direct letter frequency
- **Most powerful component** of the system
- Provides highly accurate predictions for narrow search spaces

### 3. 🤖 Deep Q-Network (DQN) Agent

**State Representation (619 dimensions):**
- 🎯 Masked word (540 dims): 20 positions × 27 one-hot features
- ✅ Guessed letters (26 dims): Binary vector
- ❤️ Lives remaining (1 dim): Normalized
- 📊 HMM probabilities (26 dims)
- 📈 Word filter probabilities (26 dims)

**Neural Network:**
```
619 → 256 → 128 → 64 → 26
```

**Training Features:**
- Experience replay buffer: 10,000 transitions
- Target network updated every 10 episodes
- Epsilon-greedy exploration: 1.0 → 0.01

### ⚡ Hybrid Strategy

```python
if matching_words <= 20:
    return word_filter_prediction()
else:
    return blend(
        word_filtering=50%,
        hmm_predictions=30%,
        dqn_q_values=20%
    )
```

---

## 📁 Project Structure

```
ML-Hackathon/
├── 📂 Data/
│   ├── corpus.txt                             # 50,000 training words
│   └── test.txt                               # 2,000 test words
├── 📓 ML_Hackathon_915_349_344_342.ipynb      # Main implementation notebook
├── 📄 Analysis_Report.pdf                     # Detailed analysis report
└── 📖 README.md                               # This file
```

---

## 🚀 Quick Start

### Setup

1. **Install dependencies:**

```bash
pip install torch tqdm matplotlib numpy
```

2. **Prepare data files:**

- Place `corpus.txt` in `Data/` directory (50,000 words)
- Place `test.txt` in `Data/` directory (2,000 words)

### Running the Notebook

The notebook `ML_Hackathon_915_349_344_342.ipynb` contains three main parts:

**Part 1: HMM Training** (~1 minute)
- Loads and preprocesses corpus
- Trains 24 HMMs (one per word length)
- Creates word matcher for filtering
- Saves models: `hangman_models.pkl`, `word_matcher.pkl`

**Part 2: RL Agent Training** (~19 minutes on GPU)
- Loads trained HMMs
- Creates Hangman environment
- Trains DQN agent for 5000 episodes
- Saves model: `dqn_agent.pth`

**Part 3: Evaluation** (~2 minutes)
- Loads all models
- Creates hybrid agent
- Evaluates on 2000 test words
- Generates visualizations
- Saves results: `evaluation_results.pkl`

### Expected Output

```
🧠 HANGMAN HMM TRAINING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Loaded 50000 words
Training HMMs for each word length...
✅ HMM Training Complete!

🤖 HANGMAN RL TRAINING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Training: 100%|██████████| 5000/5000
Final Win Rate: 7.18%
✅ Training Complete!

📊 HANGMAN EVALUATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Evaluating Hybrid Agent: 100%|██████████| 2000/2000
Success Rate: 94.40%
✅ Evaluation Complete!
```

---

## 📊 Key Findings

<table>
<tr>
<td width="50%">

### ✅ What Worked

- **🏆 Hybrid approach dominated**  
  94.40% vs 7.18% for pure RL

- **🔍 Word filtering was critical**  
  Direct corpus matching = strongest signal

- **🧠 HMM captured patterns**  
  Position-specific letter probabilities

</td>
<td width="50%">

### ❌ What Didn't Work

- **🤖 Pure RL struggled**  
  619-dimensional state space too complex

- **⚠️ Sparse rewards**  
  Mostly negative feedback hindered learning

- **🐌 Conservative exploration**  
  700 episodes to reach min epsilon

</td>
</tr>
</table>

---

## 🎯 Reward Function

| Action | Reward |
|:-------|:------:|
| ✅ Correct guess | **+10** per position revealed |
| 🎊 Win game | **+100** bonus |
| ❌ Wrong guess | **-15** penalty |
| 💀 Lose game | **-100** penalty |
| 🔁 Repeated guess | **-20** efficiency penalty |

---

## 📈 Performance by Word Length

<div align="center">

| Word Length | Win Rate | Context Level |
|:-----------:|:--------:|:-------------:|
| 2-4 letters | 50-80% | ⚠️ Limited |
| 5-9 letters | **95%+** | ✅ Optimal |
| 10-15 letters | **95%+** | ✅ Many clues |
| 16+ letters | **~100%** | 🎯 Extensive |

</div>

---

## 🔧 Training Parameters

<div align="center">

| Parameter | Value |
|:---------:|:-----:|
| 🔄 **Episodes** | 5000 |
| 📦 **Batch Size** | 64 |
| 📚 **Learning Rate** | 0.001 |
| 💰 **Gamma (Discount)** | 0.95 |
| 🎲 **Epsilon Start** | 1.0 |
| 🎯 **Epsilon Min** | 0.01 |
| 📉 **Epsilon Decay** | 0.995 |
| 💾 **Replay Buffer** | 10,000 |
| 🔄 **Target Update** | Every 10 episodes |

</div>

---

## 📝 Generated Files

| File | Description |
|:-----|:------------|
| 🧠 `hangman_models.pkl` | Trained HMM models for all word lengths |
| 🔍 `word_matcher.pkl` | Word filtering system |
| 🤖 `dqn_agent.pth` | Trained DQN agent weights |
| 📊 `training_results.png` | Training curves visualization |
| 📈 `evaluation_results.pkl` | Evaluation statistics |
| 🎨 `evaluation_results.png` | Evaluation visualizations |

---

## 💡 Key Lessons Learned

<div align="center">

### 🌟 Four Core Insights

</div>

<br>

> **1. 🧠 Domain knowledge beats pure learning**  
> Explicit word matching outperformed neural networks

> **2. 🏗️ System design matters**  
> Intelligent combination > any single method

> **3. ⚡ RL as refinement**  
> Use RL for edge cases, not learning entire strategy

> **4. ✨ Simple can be better**  
> Word filtering was simpler AND more effective

---

## 📚 References

- 🧠 Hidden Markov Models for sequence prediction
- 🤖 Deep Q-Networks (DQN) for reinforcement learning
- 💾 Experience replay and target networks
- 🎲 Epsilon-greedy exploration strategy

---

<div align="center">

### 🎓 **PES University**
**Machine Learning Hackathon 2025**

*Date: November 3, 2025*

<br>

**Made with ❤️ by Team Mohammed**

</div>
