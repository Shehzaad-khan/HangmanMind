# Hangman ML Hackathon

Intelligent Hangman agent combining Hidden Markov Models (HMM) with Reinforcement Learning (RL).

## Project Structure

```
ML-Hackathon/
├── Data/
│   ├── corpus.txt (50,000 words for training)
│   └── test.txt (2,000 words for evaluation)
├── notebooks/
│   ├── 01_hmm_preprocessing.ipynb - Data preprocessing and analysis
│   ├── 02_hmm_training.ipynb - HMM training on corpus
│   ├── 03_hmm_inference.ipynb - HMM inference testing
│   ├── 04_rl_environment.ipynb - Hangman environment implementation
│   ├── 05_rl_agent.ipynb - RL agent implementation and testing
│   ├── 06_rl_training.ipynb - RL agent training
│   ├── 07_evaluation.ipynb - Evaluation on test set
│   └── 08_analysis.ipynb - Analysis and visualization
├── src/
│   ├── hmm_model.py - HMM model implementation
│   ├── hangman_env.py - Hangman game environment
│   ├── rl_agent.py - Q-learning agent implementation
│   ├── utils.py - State encoding and utility functions
│   └── generate_report.py - PDF report generation
├── models/ - Saved trained models
├── results/ - Evaluation results and visualizations
├── Analysis_Report.pdf - Final analysis report
└── requirements.txt - Python dependencies
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure data files are in `Data/` directory:
   - `corpus.txt` (50,000 words)
   - `test.txt` (2,000 words)

## Usage

### Training

Run notebooks in order:

1. **01_hmm_preprocessing.ipynb**: Preprocess corpus data
2. **02_hmm_training.ipynb**: Train HMM model
3. **03_hmm_inference.ipynb**: Test HMM inference
4. **04_rl_environment.ipynb**: Test environment
5. **05_rl_agent.ipynb**: Test RL agent
6. **06_rl_training.ipynb**: Train RL agent (5000 episodes)
7. **07_evaluation.ipynb**: Evaluate on test set
8. **08_analysis.ipynb**: Generate visualizations and report

### Evaluation

The final evaluation runs on 2000 words from `test.txt` and calculates:
- Success Rate (win rate)
- Total Wrong Guesses
- Total Repeated Guesses
- Final Score = (Success Rate * 2000) - (Wrong Guesses * 5) - (Repeated Guesses * 2)

## Architecture

### HMM Model
- Position-aware character sequence model
- Combines position-specific frequencies, bigram/trigram probabilities, and global character frequencies
- Provides probability distribution over letters given current game state

### RL Agent
- Q-learning with epsilon-greedy exploration
- State representation: masked word + guessed letters + HMM probabilities + lives + word length
- Reward function: balances win rate with efficiency (minimizing wrong/repeated guesses)

## Results

See `Analysis_Report.pdf` for detailed results and analysis.

