# 🎮 Hangman AI Solver<div align="center"># Hangman AI Solver - HMM + Deep Reinforcement Learning



> **A hybrid approach combining Hidden Markov Models, Deep Reinforcement Learning, and Word Filtering**



## 👥 Team Members# 🎮 Hangman AI Solver## Team Members



| Name | SRN |### *Combining Hidden Markov Models + Deep Reinforcement Learning*

|------|-----|

| Mohammed Musharraf | PES2UG23CS915 || Name | SRN |

| Mohammed Shehzaad Khan | PES2U23CS349 |

| Mohammed Bilal | PES2UG23CS344 |<br>|------|-----|

| Mohammed Aahil | PES2UG23CS342 |

| Mohammed Musharraf | PES2UG23CS915 |

---

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)| Mohammed Shehzaad Khan | PES2U23CS349 |

## 🎯 Results

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)| Mohammed Bilal | PES2UG23CS344 |

| Metric | Value |

|--------|-------|[![Success Rate](https://img.shields.io/badge/Success%20Rate-94.40%25-brightgreen.svg)](.)| Mohammed Aahil | PES2UG23CS342 |

| **Success Rate** | 94.40% (2000 test games) |

| **Avg Wrong Guesses** | 2.13 per game |[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](.)

| **Repeated Guesses** | 0 |

| **Training Win Rate** (Pure RL) | 7.18% |---

| **Testing Win Rate** (Hybrid) | 94.40% |

</div>

---

## 🎯 Final Results

## 📁 Project Structure

---

```

ML-Hackathon/- **Success Rate:** 94.40% on 2000 test games

├── Data/

│   ├── corpus.txt              # 50,000 training words## 👥 Team Members- **Average Wrong Guesses:** 2.13 per game

│   └── test.txt                # 2,000 test words

├── Untitled14.ipynb            # Main implementation notebook- **Repeated Guesses:** 0

├── Analysis_Report.md          # Detailed analysis report

└── README.md                   # This file<div align="center">- **Training Win Rate (Pure RL):** 7.18%

```

- **Testing Win Rate (Hybrid System):** 94.40%

---

| Name | SRN |

## 🏗️ System Architecture

|:----:|:---:|## 📁 Project Structure

Our system uses a **three-part hybrid approach**:

| **Mohammed Musharraf** | PES2UG23CS915 |

### 1. 🧠 Hidden Markov Model (HMM)

- **States:** Position-based (0, 1, 2, ..., word_length-1)| **Mohammed Shehzaad Khan** | PES2U23CS349 |```

- **Emissions:** Letters (A-Z)

- **Models:** 24 separate HMMs for word lengths 1-24| **Mohammed Bilal** | PES2UG23CS344 |ML-Hackathon/

- **Smoothing:** Laplace smoothing (α=1.0)

- **Purpose:** Captures positional letter patterns in English words| **Mohammed Aahil** | PES2UG23CS342 |├── Data/



### 2. 🔍 Word Filtering System│   ├── corpus.txt           # 50,000 training words

- Matches current pattern against corpus words

- When ≤20 words match → uses direct letter frequency</div>│   └── test.txt             # 2,000 test words

- **Most powerful component** of the system

- Provides highly accurate predictions for narrow search spaces├── Untitled14.ipynb         # Main implementation notebook



### 3. 🤖 Deep Q-Network (DQN) Agent---├── Analysis_Report.md       # Detailed analysis report



**State Representation (619 dimensions):**└── README.md                # This file

- Masked word (540 dims): 20 positions × 27 one-hot features

- Guessed letters (26 dims): Binary vector## 🎯 Final Results```

- Lives remaining (1 dim): Normalized

- HMM probabilities (26 dims)

- Word filter probabilities (26 dims)

<div align="center">## 🏗️ Architecture

**Network Architecture:**

```

Input (619) → Dense (256) → Dense (128) → Dense (64) → Output (26)

```| Metric | Value |### Three-Part System



**Training Configuration:**|:------:|:-----:|

- Experience replay buffer: 10,000 transitions

- Target network updates: Every 10 episodes| 🏆 **Success Rate** | **94.40%** (2000 test games) |**1. Hidden Markov Model (HMM)**

- Exploration: Epsilon-greedy (1.0 → 0.01)

| 📉 **Avg Wrong Guesses** | **2.13** per game |- Position-based states (0, 1, 2, ..., word_length-1)

### ⚡ Hybrid Strategy

| 🎲 **Repeated Guesses** | **0** |- Letter emissions (A-Z)

```python

if matching_words <= 20:| 🔄 **Training Win Rate** (Pure RL) | 7.18% |- 24 separate HMMs for word lengths 1-24

    return word_filter_prediction()

else:| ✨ **Testing Win Rate** (Hybrid) | **94.40%** |- Laplace smoothing (α=1.0) for unseen patterns

    return blend(

        word_filtering=50%,- Captures positional letter patterns in English

        hmm_predictions=30%,

        dqn_q_values=20%</div>

    )

```**2. Word Filtering System**



------- Matches current pattern with corpus words



## 🚀 Quick Start- When ≤20 words match, uses direct letter frequency



### Installation## 📁 Project Structure- Most powerful component of the system



```bash- Provides accurate predictions for narrow search spaces

pip install torch tqdm matplotlib numpy

``````



### Data PreparationML-Hackathon/**3. Deep Q-Network (DQN) Agent**



Place your data files in the `Data/` directory:│- State representation: 619 dimensions

- `corpus.txt` - 50,000 training words

- `test.txt` - 2,000 test words├── 📂 Data/  - Masked word (540 dims): 20 positions × 27 one-hot features



### Running the Notebook│   ├── corpus.txt              # 50,000 training words  - Guessed letters (26 dims): Binary vector



The notebook `Untitled14.ipynb` contains three parts:│   └── test.txt                # 2,000 test words  - Lives remaining (1 dim): Normalized



**Part 1: HMM Training** (~1 minute)│  - HMM probabilities (26 dims)

- Loads and preprocesses corpus

- Trains 24 HMMs (one per word length)├── 📓 Untitled14.ipynb         # Main implementation notebook  - Word filter probabilities (26 dims)

- Creates word matcher for filtering

- Saves: `hangman_models.pkl`, `word_matcher.pkl`├── 📄 Analysis_Report.md       # Detailed analysis report- Network architecture: 619 → 256 → 128 → 64 → 26



**Part 2: RL Agent Training** (~19 minutes on GPU)└── 📖 README.md                # This file- Experience replay buffer: 10,000 transitions

- Loads trained HMMs

- Creates Hangman environment```- Target network updated every 10 episodes

- Trains DQN agent for 5000 episodes

- Saves: `dqn_agent.pth`- Epsilon-greedy exploration: 1.0 → 0.01



**Part 3: Evaluation** (~2 minutes)---

- Loads all models

- Creates hybrid agent**Hybrid Strategy**

- Evaluates on 2000 test words

- Saves: `evaluation_results.pkl`, visualizations## 🏗️ System Architecture- If matching words ≤ 20: Use word filter directly



### Expected Output- Otherwise: Blend all sources



```<div align="center">  - 50% word filtering

HANGMAN HMM TRAINING

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  - 30% HMM predictions

Loaded 50000 words

Training HMMs for each word length...### **Three-Part Hybrid System**  - 20% DQN Q-values

✅ HMM Training Complete!



HANGMAN RL TRAINING

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</div>## 🚀 Quick Start

Training: 100%|██████████| 5000/5000

Final Win Rate: 7.18%

✅ Training Complete!

<br>### Setup

HANGMAN EVALUATION

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Evaluating Hybrid Agent: 100%|██████████| 2000/2000

Success Rate: 94.40%### 🧠 **1. Hidden Markov Model (HMM)**1. **Install dependencies:**

✅ Evaluation Complete!

``````bash



---> Captures positional letter patterns in English wordspip install torch tqdm matplotlib numpy pickle



## 📊 Key Findings```



### ✅ What Worked- **States:** Position-based (0, 1, 2, ..., word_length-1)



- **Hybrid approach dominated** - 94.40% vs 7.18% for pure RL- **Emissions:** Letters (A-Z)2. **Prepare data files:**

- **Word filtering was critical** - Direct corpus matching provided strongest signal

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
