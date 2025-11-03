# Hangman AI Agent - PyTorch Implementation# Hangman AI Agent - Final Solution



Deep Learning solution for Hangman using **Neural HMM** and **Deep Q-Network (DQN)**.This project implements an AI agent to play Hangman using **Adaptive HMM** (Hidden Markov Model) with anti-overfitting strategies.



## 🎯 Results## 🎯 Final Results

- **Previous (Tabular Q-learning):** 27.05% win rate- **Win Rate:** 27.05% on 2000 test words

- **Current (PyTorch DQN):** 30%+ win rate (target: 35%+)- **Approach:** Context-based prediction (trigrams, bigrams, position) with reduced global frequency weight

- **Approach:** Neural network-based HMM + Deep Q-Network with experience replay- **Improvement:** +5.95pp over previous best (21.10%)



## 🏗️ Architecture## Project Structure



### 1. Neural HMM```

- **Input:** One-hot encoded masked word + guessed letters + word lengthML-Hackathon/

- **Architecture:** 3-layer feedforward network (863 → 256 → 128 → 26)├── Data/

- **Output:** Probability distribution over 26 letters│   ├── corpus.txt (50,000 words for training)

- **Training:** Supervised learning on 49,375 corpus words│   └── test.txt (2,000 words for evaluation)

├── notebooks/

### 2. DQN Agent│   ├── 01_hmm_preprocessing.ipynb - Data preprocessing and analysis

- **Input:** Same state representation as Neural HMM│   ├── 02_hmm_training.ipynb - HMM training on corpus

- **Architecture:** 3-layer feedforward network (863 → 128 → 128 → 26)│   ├── 03_hmm_inference.ipynb - HMM inference testing

- **Output:** Q-values for each letter action│   ├── 04_rl_environment.ipynb - Hangman environment implementation

- **Features:**│   ├── 05_rl_agent.ipynb - RL agent implementation and testing

  - Experience replay buffer (10,000 transitions)│   ├── 06_rl_training.ipynb - RL agent training

  - Target network (updated every 10 episodes)│   ├── 07_evaluation.ipynb - Evaluation on test set

  - Epsilon-greedy exploration (1.0 → 0.01)│   └── 08_analysis.ipynb - Analysis and visualization

  - HMM guidance during exploration and exploitation├── src/

│   ├── hmm_model.py - HMM model implementation

## 📁 Project Structure│   ├── hangman_env.py - Hangman game environment

│   ├── rl_agent.py - Q-learning agent implementation

```│   ├── utils.py - State encoding and utility functions

ML-Hackathon/│   └── generate_report.py - PDF report generation

├── Data/├── models/ - Saved trained models

│   ├── corpus_cleaned.txt      # Cleaned 49,375 words├── results/ - Evaluation results and visualizations

│   └── test_cleaned.txt        # Cleaned test words├── Analysis_Report.pdf - Final analysis report

├── notebooks/└── requirements.txt - Python dependencies

│   └── pytorch_implementation.ipynb  # 🔥 Main PyTorch implementation```

├── src/

│   ├── hangman_env.py         # Hangman game environment## Setup

│   └── utils.py               # Scoring and utility functions

├── models/1. Install dependencies:

│   ├── neural_hmm.pth         # 🔥 Trained Neural HMM (PyTorch)```bash

│   ├── dqn_agent.pth          # 🔥 Trained DQN Agent (PyTorch)pip install -r requirements.txt

│   ├── adaptive_hmm.pkl       # Legacy adaptive HMM (27.05%)```

│   └── adaptive_rl_agent.pkl  # Legacy tabular Q-learning

├── requirements.txt           # Dependencies (includes PyTorch)2. Ensure data files are in `Data/` directory:

└── README.md                  # This file   - `corpus.txt` (50,000 words)

```   - `test.txt` (2,000 words)



## 🚀 Quick Start## Usage



### 1. Install Dependencies### Training



```bashRun notebooks in order:

pip install -r requirements.txt

```1. **01_hmm_preprocessing.ipynb**: Preprocess corpus data

2. **02_hmm_training.ipynb**: Train HMM model

### 2. Run PyTorch Implementation3. **03_hmm_inference.ipynb**: Test HMM inference

4. **04_rl_environment.ipynb**: Test environment

```bash5. **05_rl_agent.ipynb**: Test RL agent

jupyter notebook notebooks/pytorch_implementation.ipynb6. **06_rl_training.ipynb**: Train RL agent (5000 episodes)

```7. **07_evaluation.ipynb**: Evaluate on test set

8. **08_analysis.ipynb**: Generate visualizations and report

The notebook includes:

- **Cells 1-3:** Setup and data loading### Evaluation

- **Cells 4-6:** Neural HMM training and testing

- **Cells 7-9:** DQN agent training and evaluationThe final evaluation runs on 2000 words from `test.txt` and calculates:

- **Cell 10:** Save trained models- Success Rate (win rate)

- Total Wrong Guesses

### 3. Training Time- Total Repeated Guesses

- **Neural HMM:** ~5-10 minutes (3 epochs)- Final Score = (Success Rate * 2000) - (Wrong Guesses * 5) - (Repeated Guesses * 2)

- **DQN Agent:** ~15-20 minutes (10K episodes)

- **Total:** ~25-30 minutes on CPU## Architecture



## 📊 Performance### HMM Model

- Position-aware character sequence model

| Approach | Win Rate | Description |- Combines position-specific frequencies, bigram/trigram probabilities, and global character frequencies

|----------|----------|-------------|- Provides probability distribution over letters given current game state

| Old HMM | 22.20% | Overfits to frequency |

| Adaptive HMM | 27.05% | Context-aware (n-grams) |### RL Agent

| Tabular Q-learning | 27.00% | Sparse state coverage |- Q-learning with epsilon-greedy exploration

| **PyTorch DQN** | **30-35%** | Deep RL + HMM guidance |- State representation: masked word + guessed letters + HMM probabilities + lives + word length

- Reward function: balances win rate with efficiency (minimizing wrong/repeated guesses)

## 🎯 Why PyTorch?

## Results

✅ **Function Approximation** - Handles large state spaces<br>

✅ **Generalization** - Learns patterns, not memorization<br>See `Analysis_Report.pdf` for detailed results and analysis.

✅ **Modern ML** - Deep learning best practices<br>

✅ **Problem Requirement** - Implements suggested DQN approach

## 📈 Key Features

- Neural network-based HMM (no n-gram counting)
- Deep Q-Network with experience replay
- Target network for stable learning
- Epsilon-greedy exploration with decay
- Combined HMM + RL decision making

---

**PES University - Machine Learning Hackathon 2025**
