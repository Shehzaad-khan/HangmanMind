# Hangman AI Solver 🎮# 🎮 Hangman AI Solver<div align="center"># Hangman AI Solver - HMM + Deep Reinforcement Learning



A sophisticated AI agent that plays Hangman with a 94.40% success rate by combining Hidden Markov Models, Deep Reinforcement Learning, and intelligent word filtering.



## Team Members> **A hybrid approach combining Hidden Markov Models, Deep Reinforcement Learning, and Word Filtering**



| Name | SRN |

|------|-----|

| Mohammed Musharraf | PES2UG23CS915 |## 👥 Team Members# 🎮 Hangman AI Solver## Team Members

| Mohammed Shehzaad Khan | PES2U23CS349 |

| Mohammed Bilal | PES2UG23CS344 |

| Mohammed Aahil | PES2UG23CS342 |

| Name | SRN |### *Combining Hidden Markov Models + Deep Reinforcement Learning*

## Overview

|------|-----|

This project implements a hybrid AI system that combines three complementary approaches to achieve exceptional performance in the Hangman word-guessing game:

| Mohammed Musharraf | PES2UG23CS915 || Name | SRN |

- **Hidden Markov Models (HMM)**: Learn position-specific letter patterns from a corpus of 50,000 words

- **Deep Q-Network (DQN)**: Reinforcement learning agent that learns strategic decision-making| Mohammed Shehzaad Khan | PES2U23CS349 |

- **Word Filtering**: Pattern matching against the corpus for highly accurate predictions

| Mohammed Bilal | PES2UG23CS344 |<br>|------|-----|

## Results

| Mohammed Aahil | PES2UG23CS342 |

Our hybrid system achieved outstanding performance on 2,000 test games:

| Mohammed Musharraf | PES2UG23CS915 |

- **Success Rate**: 94.40%

- **Average Wrong Guesses**: 2.13 per game---

- **Repeated Guesses**: 0

- **Improvement over Pure RL**: 94.40% vs 7.18% (13x better)[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)| Mohammed Shehzaad Khan | PES2U23CS349 |



## Architecture## 🎯 Results



### 1. Hidden Markov Model (HMM)[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)| Mohammed Bilal | PES2UG23CS344 |



The HMM component captures positional letter distributions in English words.| Metric | Value |



**Key Features:**|--------|-------|[![Success Rate](https://img.shields.io/badge/Success%20Rate-94.40%25-brightgreen.svg)](.)| Mohammed Aahil | PES2UG23CS342 |

- 24 separate models trained for word lengths 1-24

- Position-based states (0, 1, 2, ..., word_length-1)| **Success Rate** | 94.40% (2000 test games) |

- Letter emissions (A-Z)

- Laplace smoothing (α=1.0) for unseen patterns| **Avg Wrong Guesses** | 2.13 per game |[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](.)



**How it works:**| **Repeated Guesses** | 0 |

- Analyzes where specific letters typically appear in words of a given length

- Provides probability distributions for the next letter guess based on revealed positions| **Training Win Rate** (Pure RL) | 7.18% |---

- Example: 'E' is more likely at the end of words, 'S' at the beginning of plurals

| **Testing Win Rate** (Hybrid) | 94.40% |

### 2. Word Filtering System

</div>

The most powerful component that performs direct pattern matching.

---

**Strategy:**

- Filters corpus to match the current masked word pattern## 🎯 Final Results

- When ≤20 words match, uses direct letter frequency from matching words

- Highly accurate for constrained search spaces## 📁 Project Structure

- Example: Pattern `"a_ple"` → matches "apple", "ample" → suggests 'p'

---

### 3. Deep Q-Network (DQN)

```

Reinforcement learning agent that learns to combine all available signals.

ML-Hackathon/- **Success Rate:** 94.40% on 2000 test games

**State Space (619 dimensions):**

- Masked word representation (540 dims): 20 positions × 27 features (26 letters + blank)├── Data/

- Guessed letters (26 dims): Binary vector indicating which letters were tried

- Lives remaining (1 dim): Normalized count of remaining incorrect guesses│   ├── corpus.txt              # 50,000 training words## 👥 Team Members- **Average Wrong Guesses:** 2.13 per game

- HMM predictions (26 dims): Probability distribution from HMM

- Word filter predictions (26 dims): Probability distribution from word matching│   └── test.txt                # 2,000 test words



**Network Architecture:**├── Untitled14.ipynb            # Main implementation notebook- **Repeated Guesses:** 0

```

Input Layer (619) ├── Analysis_Report.md          # Detailed analysis report

    ↓

Dense Layer (256) + ReLU└── README.md                   # This file<div align="center">- **Training Win Rate (Pure RL):** 7.18%

    ↓

Dense Layer (128) + ReLU```

    ↓

Dense Layer (64) + ReLU- **Testing Win Rate (Hybrid System):** 94.40%

    ↓

Output Layer (26)---

```

| Name | SRN |

**Training Details:**

- Episodes: 5,000## 🏗️ System Architecture

- Experience replay buffer: 10,000 transitions

- Epsilon-greedy exploration: 1.0 → 0.01 (decay: 0.995)|:----:|:---:|## 📁 Project Structure

- Target network update: Every 10 episodes

- Optimizer: Adam (lr=0.001)Our system uses a **three-part hybrid approach**:

- Discount factor (γ): 0.95

- Training time: ~19 minutes on GPU| **Mohammed Musharraf** | PES2UG23CS915 |



### Hybrid Decision Strategy### 1. 🧠 Hidden Markov Model (HMM)



The system intelligently combines all three components:- **States:** Position-based (0, 1, 2, ..., word_length-1)| **Mohammed Shehzaad Khan** | PES2U23CS349 |```



```python- **Emissions:** Letters (A-Z)

if len(matching_words) <= 20:

    # Use direct word filtering for high confidence- **Models:** 24 separate HMMs for word lengths 1-24| **Mohammed Bilal** | PES2UG23CS344 |ML-Hackathon/

    return word_filter_prediction()

else:- **Smoothing:** Laplace smoothing (α=1.0)

    # Blend all sources with weighted combination

    prediction = (- **Purpose:** Captures positional letter patterns in English words| **Mohammed Aahil** | PES2UG23CS342 |├── Data/

        0.50 * word_filter_probabilities +

        0.30 * hmm_probabilities +

        0.20 * dqn_q_values

    )### 2. 🔍 Word Filtering System│   ├── corpus.txt           # 50,000 training words

    return best_unguessed_letter(prediction)

```- Matches current pattern against corpus words



## Project Structure- When ≤20 words match → uses direct letter frequency</div>│   └── test.txt             # 2,000 test words



```- **Most powerful component** of the system

ML-Hackathon/

├── Data/- Provides highly accurate predictions for narrow search spaces├── Untitled14.ipynb         # Main implementation notebook

│   ├── corpus.txt              # 50,000 training words

│   └── test.txt                # 2,000 test words

├── Untitled14.ipynb            # Main implementation notebook

├── Analysis_Report.md          # Detailed project analysis### 3. 🤖 Deep Q-Network (DQN) Agent---├── Analysis_Report.md       # Detailed analysis report

└── README.md                   # This file

```



## Getting Started**State Representation (619 dimensions):**└── README.md                # This file



### Prerequisites- Masked word (540 dims): 20 positions × 27 one-hot features



```bash- Guessed letters (26 dims): Binary vector## 🎯 Final Results```

pip install torch tqdm matplotlib numpy

```- Lives remaining (1 dim): Normalized



### Data Setup- HMM probabilities (26 dims)



Ensure you have the following files in the `Data/` directory:- Word filter probabilities (26 dims)

- `corpus.txt` - Training corpus (50,000 words)

- `test.txt` - Test set (2,000 words)<div align="center">## 🏗️ Architecture



### Running the Notebook**Network Architecture:**



The `Untitled14.ipynb` notebook is divided into three sequential parts:```



#### Part 1: HMM Training (~1 minute)Input (619) → Dense (256) → Dense (128) → Dense (64) → Output (26)



Trains the Hidden Markov Models on the corpus.```| Metric | Value |### Three-Part System



**Steps:**

1. Loads and preprocesses `corpus.txt`

2. Trains 24 separate HMMs for different word lengths**Training Configuration:**|:------:|:-----:|

3. Creates word matcher for pattern filtering

4. Saves models to `hangman_models.pkl` and `word_matcher.pkl`- Experience replay buffer: 10,000 transitions



**Output:**- Target network updates: Every 10 episodes| 🏆 **Success Rate** | **94.40%** (2000 test games) |**1. Hidden Markov Model (HMM)**

```

HANGMAN HMM TRAINING- Exploration: Epsilon-greedy (1.0 → 0.01)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Loaded 50000 words| 📉 **Avg Wrong Guesses** | **2.13** per game |- Position-based states (0, 1, 2, ..., word_length-1)

Training HMMs for each word length...

✅ HMM Training Complete!### ⚡ Hybrid Strategy

```

| 🎲 **Repeated Guesses** | **0** |- Letter emissions (A-Z)

#### Part 2: DQN Training (~19 minutes on GPU)

```python

Trains the reinforcement learning agent.

if matching_words <= 20:| 🔄 **Training Win Rate** (Pure RL) | 7.18% |- 24 separate HMMs for word lengths 1-24

**Steps:**

1. Loads trained HMM models    return word_filter_prediction()

2. Creates Hangman environment with reward shaping

3. Trains DQN agent for 5,000 episodeselse:| ✨ **Testing Win Rate** (Hybrid) | **94.40%** |- Laplace smoothing (α=1.0) for unseen patterns

4. Saves trained agent to `dqn_agent.pth`

    return blend(

**Output:**

```        word_filtering=50%,- Captures positional letter patterns in English

HANGMAN RL TRAINING

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━        hmm_predictions=30%,

Training: 100%|██████████| 5000/5000

Episode 5000/5000 | Win Rate: 7.18% | Epsilon: 0.01        dqn_q_values=20%</div>

✅ Training Complete!

```    )



#### Part 3: Evaluation (~2 minutes)```**2. Word Filtering System**



Evaluates the complete hybrid system.



**Steps:**------- Matches current pattern with corpus words

1. Loads all trained models

2. Creates hybrid agent combining HMM, DQN, and word filtering

3. Evaluates on 2,000 test words

4. Generates performance visualizations## 🚀 Quick Start- When ≤20 words match, uses direct letter frequency

5. Saves results to `evaluation_results.pkl`



**Output:**

```### Installation## 📁 Project Structure- Most powerful component of the system

HANGMAN EVALUATION

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Evaluating Hybrid Agent: 100%|██████████| 2000/2000

Success Rate: 94.40%```bash- Provides accurate predictions for narrow search spaces

Average Wrong Guesses: 2.13

✅ Evaluation Complete!pip install torch tqdm matplotlib numpy

```

``````

## Performance Analysis



### What Worked

### Data PreparationML-Hackathon/**3. Deep Q-Network (DQN) Agent**

✅ **Hybrid approach was crucial**

- Pure RL achieved only 7.18% success

- Hybrid system reached 94.40% (13x improvement)

- Different components complemented each other's weaknessesPlace your data files in the `Data/` directory:│- State representation: 619 dimensions



✅ **Word filtering dominated**- `corpus.txt` - 50,000 training words

- Most powerful individual component

- Direct pattern matching provided the strongest signal- `test.txt` - 2,000 test words├── 📂 Data/  - Masked word (540 dims): 20 positions × 27 one-hot features

- Especially effective when search space narrowed down



✅ **HMM captured language structure**

- Position-specific letter distributions worked well### Running the Notebook│   ├── corpus.txt              # 50,000 training words  - Guessed letters (26 dims): Binary vector

- Helped in early game when pattern matching wasn't sufficient

- Complemented word filtering nicely



### What Didn't WorkThe notebook `Untitled14.ipynb` contains three parts:│   └── test.txt                # 2,000 test words  - Lives remaining (1 dim): Normalized



❌ **Pure RL struggled significantly**

- Only 7.18% win rate after 5,000 episodes

- 619-dimensional state space was challenging**Part 1: HMM Training** (~1 minute)│  - HMM probabilities (26 dims)

- Required extensive training to learn basic patterns

- Loads and preprocesses corpus

❌ **Sparse reward problem**

- Mostly negative feedback during training- Trains 24 HMMs (one per word length)├── 📓 Untitled14.ipynb         # Main implementation notebook  - Word filter probabilities (26 dims)

- Win bonus came too late to effectively guide learning

- Agent struggled to discover good strategies- Creates word matcher for filtering



❌ **Conservative exploration schedule**- Saves: `hangman_models.pkl`, `word_matcher.pkl`├── 📄 Analysis_Report.md       # Detailed analysis report- Network architecture: 619 → 256 → 128 → 64 → 26

- Took 700 episodes to reach minimum epsilon

- Could have explored more aggressively early on

- Faster decay might have improved learning

**Part 2: RL Agent Training** (~19 minutes on GPU)└── 📖 README.md                # This file- Experience replay buffer: 10,000 transitions

### Performance by Word Length

- Loads trained HMMs

| Word Length | Success Rate | Reasoning |

|-------------|--------------|-----------|- Creates Hangman environment```- Target network updated every 10 episodes

| 2-4 letters | 50-80% | Limited context, fewer possible patterns |

| 5-9 letters | 95%+ | Optimal range with enough context |- Trains DQN agent for 5000 episodes

| 10-15 letters | 95%+ | Multiple clues available |

| 16+ letters | ~100% | Extensive context makes guessing easy |- Saves: `dqn_agent.pth`- Epsilon-greedy exploration: 1.0 → 0.01



## Reward Function



The agent uses a carefully designed reward structure:**Part 3: Evaluation** (~2 minutes)---



| Event | Reward | Rationale |- Loads all models

|-------|--------|-----------|

| Correct guess | +10 per revealed position | Encourages productive guesses |- Creates hybrid agent**Hybrid Strategy**

| Win game | +100 | Strong positive reinforcement |

| Wrong guess | -15 | Discourages random guessing |- Evaluates on 2000 test words

| Lose game | -100 | Strong negative feedback |

| Repeated guess | -20 | Heavily penalizes inefficiency |- Saves: `evaluation_results.pkl`, visualizations## 🏗️ System Architecture- If matching words ≤ 20: Use word filter directly



## Key Insights



### 1. Domain Knowledge Beats Pure Learning### Expected Output- Otherwise: Blend all sources



Explicit word matching with the corpus significantly outperformed the neural network approach. Sometimes, simple rule-based systems can be more effective than complex learning algorithms.



### 2. System Design is Critical```<div align="center">  - 50% word filtering



The intelligent combination of multiple techniques proved more powerful than any single method. The hybrid approach leveraged the strengths of each component:HANGMAN HMM TRAINING

- Word filtering for accuracy

- HMM for language patterns  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  - 30% HMM predictions

- RL for adaptive decision-making

Loaded 50000 words

### 3. RL as Refinement, Not Foundation

Training HMMs for each word length...### **Three-Part Hybrid System**  - 20% DQN Q-values

Deep RL worked best as a refinement tool rather than the primary strategy. Use RL to handle edge cases and combine multiple signals, not to learn the entire task from scratch.

✅ HMM Training Complete!

### 4. Simple Can Be Better



The straightforward word filtering approach was both simpler to implement and more effective than the complex neural network. Don't overcomplicate solutions without evidence of benefit.

HANGMAN RL TRAINING

## Generated Files

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</div>## 🚀 Quick Start

After running all three parts of the notebook, you'll have:

Training: 100%|██████████| 5000/5000

| File | Description |

|------|-------------|Final Win Rate: 7.18%

| `hangman_models.pkl` | Trained HMM models for all word lengths |

| `word_matcher.pkl` | Word filtering system with corpus index |✅ Training Complete!

| `dqn_agent.pth` | Trained DQN agent weights |

| `training_results.png` | Training curves and statistics |<br>### Setup

| `evaluation_results.pkl` | Detailed evaluation metrics |

| `evaluation_results.png` | Performance visualizations |HANGMAN EVALUATION



## Technical References━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━



This project implements concepts from:Evaluating Hybrid Agent: 100%|██████████| 2000/2000

- **Hidden Markov Models**: Probabilistic sequence modeling

- **Deep Q-Networks**: Value-based reinforcement learning [(Mnih et al., 2015)](https://www.nature.com/articles/nature14236)Success Rate: 94.40%### 🧠 **1. Hidden Markov Model (HMM)**1. **Install dependencies:**

- **Experience Replay**: Stabilizing RL training [(Lin, 1992)](https://link.springer.com/article/10.1007/BF00992699)

- **Target Networks**: Reducing Q-value overestimation✅ Evaluation Complete!

- **Epsilon-Greedy Exploration**: Balancing exploration and exploitation

``````bash

## Future Improvements



Potential enhancements for even better performance:

---> Captures positional letter patterns in English wordspip install torch tqdm matplotlib numpy pickle

1. **Prioritized Experience Replay**: Sample important transitions more frequently

2. **Double DQN**: Reduce overestimation bias in Q-values

3. **Dueling Network Architecture**: Separate value and advantage streams

4. **Letter Frequency Priors**: Incorporate English letter frequency statistics## 📊 Key Findings```

5. **Curriculum Learning**: Train on easier words first, gradually increase difficulty

6. **Ensemble Methods**: Combine multiple DQN agents for robustness



## License### ✅ What Worked- **States:** Position-based (0, 1, 2, ..., word_length-1)



This project was developed for educational purposes as part of the PES University Machine Learning Hackathon 2025.



---- **Hybrid approach dominated** - 94.40% vs 7.18% for pure RL- **Emissions:** Letters (A-Z)2. **Prepare data files:**



**PES University - Machine Learning Hackathon 2025**  - **Word filtering was critical** - Direct corpus matching provided strongest signal

**Date:** November 3, 2025

- **HMM captured patterns** - Position-specific letter probabilities worked well- **Models:** 24 separate HMMs for word lengths 1-24- Place `corpus.txt` in root directory (50,000 words)



### ❌ What Didn't Work- **Smoothing:** Laplace smoothing (α=1.0) for unseen patterns- Place `test_words.txt` in root directory (2,000 words)



- **Pure RL struggled** - 619-dimensional state space made learning difficult- **Purpose:** Learn where letters typically appear in words

- **Sparse rewards** - Mostly negative feedback hindered learning

- **Conservative exploration** - 700 episodes to reach minimum epsilon was too slow### Running the Notebook



---<br>



## 🎯 Reward FunctionThe notebook `Untitled14.ipynb` contains three main parts:



| Action | Reward |### 🔍 **2. Word Filtering System**

|--------|--------|

| Correct guess | +10 per position revealed |**Part 1: HMM Training**

| Win game | +100 bonus |

| Wrong guess | -15 penalty |> Direct pattern matching with corpus- Loads and preprocesses corpus

| Lose game | -100 penalty |

| Repeated guess | -20 efficiency penalty |- Trains 24 HMMs (one per word length)



---- Matches current pattern against corpus words- Creates word matcher for filtering



## 📈 Performance by Word Length- When ≤20 words match → uses direct letter frequency- Saves models: `hangman_models.pkl`, `word_matcher.pkl`



| Word Length | Win Rate | Context |- **Most powerful component** of the system- Training time: ~1 minute

|-------------|----------|---------|

| 2-4 letters | 50-80% | Limited context |- Provides highly accurate predictions for narrow search spaces

| 5-9 letters | 95%+ | Optimal range |

| 10-15 letters | 95%+ | Many clues |**Part 2: RL Agent Training**

| 16+ letters | ~100% | Extensive context |

<br>- Loads trained HMMs

---

- Creates Hangman environment

## 🔧 Training Parameters

### 🤖 **3. Deep Q-Network (DQN) Agent**- Trains DQN agent for 5000 episodes

| Parameter | Value |

|-----------|-------|- Saves model: `dqn_agent.pth`

| Episodes | 5000 |

| Batch Size | 64 |> Reinforcement learning for strategic decision-making- Training time: ~19 minutes on GPU

| Learning Rate | 0.001 |

| Gamma (Discount) | 0.95 |

| Epsilon Start | 1.0 |

| Epsilon Min | 0.01 |**State Representation (619 dimensions):****Part 3: Evaluation**

| Epsilon Decay | 0.995 |

| Replay Buffer | 10,000 |- 🎯 Masked word (540 dims): 20 positions × 27 one-hot features- Loads all models

| Target Update | Every 10 episodes |

- ✅ Guessed letters (26 dims): Binary vector- Creates hybrid agent

---

- ❤️ Lives remaining (1 dim): Normalized- Evaluates on 2000 test words

## 📝 Generated Files

- 📊 HMM probabilities (26 dims)- Generates visualizations

- `hangman_models.pkl` - Trained HMM models for all word lengths

- `word_matcher.pkl` - Word filtering system- 📈 Word filter probabilities (26 dims)- Saves results: `evaluation_results.pkl`

- `dqn_agent.pth` - Trained DQN agent weights

- `training_results.png` - Training curves visualization- Evaluation time: ~2 minutes

- `evaluation_results.pkl` - Evaluation statistics

- `evaluation_results.png` - Evaluation visualizations**Neural Network:**



---```### Expected Output



## 💡 Key Lessons619 → 256 → 128 → 64 → 26



1. **Domain knowledge beats pure learning** - Explicit word matching outperformed neural networks``````

2. **System design matters** - Intelligent combination of techniques > any single method

3. **RL as refinement** - Use RL to handle edge cases, not learn entire strategyHANGMAN HMM TRAINING

4. **Simple can be better** - Word filtering was simpler AND more effective than complex neural networks

**Training Features:**Loaded 50000 words

---

- Experience replay buffer: 10,000 transitionsTraining HMMs for each word length...

## 📚 References

- Target network updates: Every 10 episodes✅ HMM Training Complete!

- Hidden Markov Models for sequence prediction

- Deep Q-Networks (DQN) for reinforcement learning- Exploration: Epsilon-greedy (1.0 → 0.01)

- Experience replay and target networks for stable learning

- Epsilon-greedy exploration strategyHANGMAN RL TRAINING



---<br>Training: 100%|██████████| 5000/5000



**PES University - Machine Learning Hackathon 2025**  Final Win Rate: 7.18%

*November 3, 2025*

### ⚡ **Hybrid Strategy**✅ Training Complete!



```pythonHANGMAN EVALUATION

if matching_words <= 20:Evaluating Hybrid Agent: 100%|██████████| 2000/2000

    return word_filter_prediction()Success Rate: 94.40%

else:✅ Evaluation Complete!

    return blend(```

        word_filtering=50%,

        hmm_predictions=30%,## 📊 Key Findings

        dqn_q_values=20%

    )### What Worked

```

✅ **Hybrid approach dominated** - Combining word filtering, HMM, and RL achieved 94.40% vs 7.18% for pure RL

---

✅ **Word filtering was critical** - Direct corpus matching provided the strongest signal

## 🚀 Quick Start Guide

✅ **HMM captured language patterns** - Position-specific letter probabilities worked well

### 📦 **Installation**

### What Didn't Work

```bash

pip install torch tqdm matplotlib numpy❌ **Pure RL struggled** - 619-dimensional state space made learning difficult

```

❌ **Sparse rewards** - Mostly negative feedback hindered learning

### 📂 **Data Preparation**

❌ **Too conservative exploration** - 700 episodes to reach minimum epsilon was too slow

Place your data files in the `Data/` directory:

- `corpus.txt` - 50,000 training words## 🎯 Reward Function

- `test.txt` - 2,000 test words

- **Correct guess:** +10 per position revealed + 100 if won

### ▶️ **Running the Notebook**- **Wrong guess:** -15 (+ -100 if lost)

- **Repeated guess:** -20 (heavy efficiency penalty)

The notebook `Untitled14.ipynb` has three main sections:

## � Performance by Word Length

<br>

- **Short words (2-4 letters):** ~50-80% win rate (limited context)

#### **Part 1: 🧠 HMM Training** (~1 minute)- **Medium words (5-9 letters):** ~95%+ win rate (optimal)

- Loads and preprocesses corpus- **Long words (10-15 letters):** ~95%+ win rate (many clues)

- Trains 24 HMMs (one per word length)- **Very long words (16+ letters):** ~100% win rate (extensive context)

- Creates word matcher for filtering

- **Saves:** `hangman_models.pkl`, `word_matcher.pkl`## 🔧 Training Parameters



#### **Part 2: 🤖 RL Agent Training** (~19 minutes on GPU)| Parameter | Value |

- Loads trained HMMs|-----------|-------|

- Creates Hangman environment| Episodes | 5000 |

- Trains DQN agent for 5000 episodes| Batch Size | 64 |

- **Saves:** `dqn_agent.pth`| Learning Rate | 0.001 |

| Gamma (Discount) | 0.95 |

#### **Part 3: 📊 Evaluation** (~2 minutes)| Epsilon Start | 1.0 |

- Loads all models| Epsilon Min | 0.01 |

- Creates hybrid agent| Epsilon Decay | 0.995 |

- Evaluates on 2000 test words| Replay Buffer | 10,000 |

- **Saves:** `evaluation_results.pkl`, visualizations| Target Update | Every 10 episodes |



<br>## 📝 Files Generated



### 📺 **Expected Output**- `hangman_models.pkl` - Trained HMM models for all word lengths

- `word_matcher.pkl` - Word filtering system

```- `dqn_agent.pth` - Trained DQN agent weights

🧠 HANGMAN HMM TRAINING- `training_results.png` - Training curves visualization

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━- `evaluation_results.pkl` - Evaluation statistics

Loaded 50000 words- `evaluation_results.png` - Evaluation visualizations

Training HMMs for each word length...

✅ HMM Training Complete!## � Key Lessons



🤖 HANGMAN RL TRAINING1. **Domain knowledge beats pure learning** - Explicit word matching outperformed neural networks

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━2. **System design matters** - Intelligent combination of techniques is more powerful than any single method

Training: 100%|██████████| 5000/50003. **RL as refinement** - Use RL to handle edge cases, not learn entire strategy from scratch

Final Win Rate: 7.18%4. **Simple can be better** - Word filtering was simpler and more effective than complex neural networks

✅ Training Complete!

## � References

📊 HANGMAN EVALUATION

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━- Hidden Markov Models for sequence prediction

Evaluating Hybrid Agent: 100%|██████████| 2000/2000- Deep Q-Networks (DQN) for reinforcement learning

Success Rate: 94.40%- Experience replay and target networks for stable learning

✅ Evaluation Complete!- Epsilon-greedy exploration strategy

```

---

---

**PES University - Machine Learning Hackathon 2025**  

## 📊 Key Findings**Date:** November 3, 2025


<table>
<tr>
<td width="50%">

### ✅ **What Worked**

- **🏆 Hybrid approach dominated**  
  94.40% vs 7.18% for pure RL

- **🔍 Word filtering was critical**  
  Direct corpus matching = strongest signal

- **🧠 HMM captured patterns**  
  Position-specific letter probabilities

</td>
<td width="50%">

### ❌ **What Didn't Work**

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

### 🌟 **Four Core Insights**

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
