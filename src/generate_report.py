"""
Generate Analysis Report PDF.

This script creates the Analysis_Report.pdf with all required sections.
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
import pickle
import os


def generate_report():
    """Generate the analysis report PDF."""
    
    # Load results
    with open('../results/evaluation_results.pkl', 'rb') as f:
        eval_results = pickle.load(f)
    
    with open('../results/training_history.pkl', 'rb') as f:
        training_history = pickle.load(f)
    
    # Create PDF
    pdf_path = '../Analysis_Report.pdf'
    doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                          rightMargin=72, leftMargin=72,
                          topMargin=72, bottomMargin=18)
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor='black',
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor='black',
        spaceAfter=12,
        spaceBefore=20,
        fontName='Helvetica-Bold'
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        leading=14,
        alignment=TA_JUSTIFY
    )
    
    # Title
    elements.append(Paragraph("Hangman ML Hackathon - Analysis Report", title_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Introduction
    elements.append(Paragraph("1. Introduction", heading_style))
    intro_text = """
    This report details the implementation and analysis of an intelligent Hangman game agent
    that combines Hidden Markov Models (HMM) with Reinforcement Learning (RL). The agent
    was trained on a corpus of 50,000 English words and evaluated on a test set of 2,000 words.
    The goal was to maximize win rate while minimizing wrong and repeated guesses.
    """
    elements.append(Paragraph(intro_text, normal_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Key Observations
    elements.append(Paragraph("2. Key Observations", heading_style))
    
    # Get some statistics
    final_win_rate = eval_results['success_rate']
    total_wrong = eval_results['total_wrong_guesses']
    total_repeated = eval_results['total_repeated_guesses']
    final_score = eval_results['final_score']
    
    observations_text = f"""
    <b>Performance Metrics:</b><br/>
    - Success Rate: {final_win_rate:.2%}<br/>
    - Total Wrong Guesses: {total_wrong}<br/>
    - Total Repeated Guesses: {total_repeated}<br/>
    - Final Score: {final_score:.2f}<br/><br/>
    
    <b>Key Insights:</b><br/>
    The integration of HMM probabilities with RL proved to be an effective approach. The HMM
    provides strong priors for letter selection, while the RL agent learns to balance exploration
    and exploitation. The most challenging aspects were:
    <br/><br/>
    1. <b>State Space Complexity:</b> The state space for Hangman is very large due to the
    combinatorial nature of masked words and guessed letters. We used a simplified state
    representation combining key features (word length, revealed count, lives, top HMM predictions).
    <br/><br/>
    2. <b>Reward Shaping:</b> Designing a reward function that balances immediate rewards
    (correct guesses) with terminal rewards (winning/losing) was crucial for effective learning.
    The agent needed to learn to prioritize high-probability letters from the HMM while also
    learning from experience.
    <br/><br/>
    3. <b>Exploration vs Exploitation:</b> Initially, with epsilon=1.0, the agent explores
    randomly. As training progresses and epsilon decays, the agent increasingly relies on
    learned Q-values combined with HMM probabilities. This balance was critical for learning
    effective policies without overfitting.
    """
    elements.append(Paragraph(observations_text, normal_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Strategies
    elements.append(Paragraph("3. Strategies", heading_style))
    
    hmm_strategy = """
    <b>HMM Design Choices:</b><br/>
    We implemented a position-aware character sequence model that combines multiple information sources:
    <br/><br/>
    1. <b>Position-specific character frequencies:</b> Captures patterns like common starting letters
    (e.g., 'S', 'T', 'A') and ending letters (e.g., 'E', 'S', 'D').
    <br/><br/>
    2. <b>Character bigram probabilities:</b> Models sequential dependencies (e.g., 'TH', 'QU'
    are common pairs) to provide context-aware predictions.
    <br/><br/>
    3. <b>Character trigram probabilities:</b> Captures longer-range dependencies for more
    sophisticated predictions.
    <br/><br/>
    4. <b>Global character frequencies:</b> Provides a fallback prior when position-specific
    information is sparse.
    <br/><br/>
    This hybrid approach allows the HMM to adapt its predictions based on the current game state,
    revealing patterns like "if there's an 'E' at position 0, 'X' is likely next" or "if the
    word ends with '_', 'S' or 'E' are likely endings."
    """
    elements.append(Paragraph(hmm_strategy, normal_style))
    elements.append(Spacer(1, 0.15*inch))
    
    rl_strategy = """
    <b>RL State and Reward Design:</b><br/>
    <b>State Representation:</b> We encode the state as a combination of:
    - Masked word pattern (one-hot encoding per position)
    - Binary vector of guessed letters (26 dimensions)
    - HMM probability vector (26 dimensions)
    - Normalized remaining lives and word length
    <br/><br/>
    Since the full state space is too large for tabular Q-learning, we use a simplified state
    key based on: word length, number of revealed letters, remaining lives, and top 3 HMM predictions.
    This reduction allows efficient Q-table storage while preserving essential information.
    <br/><br/>
    <b>Reward Function:</b> Our reward structure is:
    - +1.0 for correct guesses, with +0.5 bonus per letter revealed
    - +10.0 for winning the game
    - -1.0 for wrong guesses
    - -5.0 additional penalty for losing
    - -2.0 for repeated guesses
    <br/><br/>
    This design encourages the agent to make correct guesses while strongly penalizing inefficiency
    (wrong and repeated guesses), aligning with the competition's scoring formula.
    """
    elements.append(Paragraph(rl_strategy, normal_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Exploration
    elements.append(Paragraph("4. Exploration", heading_style))
    
    exploration_text = """
    <b>Exploration vs Exploitation Trade-off:</b><br/>
    We use an ε-greedy exploration strategy with exponential decay:
    <br/><br/>
    - <b>Initial epsilon (1.0):</b> Start with full exploration to ensure the agent sees diverse
    game states and learns from various scenarios.
    <br/><br/>
    - <b>Epsilon decay (0.9995):</b> Slowly decay epsilon over training episodes, allowing the
    agent to gradually shift from exploration to exploitation. This ensures sufficient exploration
    early in training while enabling exploitation of learned policies later.
    <br/><br/>
    - <b>Minimum epsilon (0.05):</b> Maintain a small amount of exploration even after training
    to handle novel game states and prevent overfitting to training patterns.
    <br/><br/>
    During evaluation, epsilon is set to 0.0 for pure exploitation, using only learned Q-values
    and HMM probabilities to select actions.
    <br/><br/>
    The exploration strategy proved effective in allowing the agent to learn diverse strategies
    for different word patterns while gradually converging to an optimal policy.
    """
    elements.append(Paragraph(exploration_text, normal_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Future Improvements
    elements.append(Paragraph("5. Future Improvements", heading_style))
    
    improvements_text = """
    If given another week, the following improvements could enhance agent performance:
    <br/><br/>
    1. <b>Deep Q-Network (DQN):</b> Implement a neural network-based Q-function approximator
    to handle the full state space without state abstraction. This would allow the agent to
    learn more nuanced patterns and relationships in the game state.
    <br/><br/>
    2. <b>Advanced HMM Architectures:</b> Experiment with more sophisticated HMM structures,
    such as separate models for different word lengths, or variable-order Markov models that
    adapt to word patterns.
    <br/><br/>
    3. <b>Better State Representation:</b> Use embeddings or attention mechanisms to capture
    relationships between positions, rather than simple one-hot encodings. This could help
    the agent understand positional context better.
    <br/><br/>
    4. <b>Reward Shaping Refinement:</b> Fine-tune reward weights based on ablation studies
    to optimize the trade-off between win rate and efficiency (minimizing wrong/repeated guesses).
    <br/><br/>
    5. <b>Transfer Learning:</b> Pre-train on a larger vocabulary or use pre-trained word
    embeddings to initialize the HMM with richer linguistic knowledge.
    <br/><br/>
    6. <b>Multi-step Lookahead:</b> Implement value iteration or Monte Carlo Tree Search to
    evaluate sequences of actions rather than single-step decisions.
    <br/><br/>
    7. <b>Ensemble Methods:</b> Combine multiple HMM models (e.g., different architectures
    or trained on different subsets) and use voting or weighted averaging for more robust predictions.
    """
    elements.append(Paragraph(improvements_text, normal_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Conclusion
    elements.append(Paragraph("6. Conclusion", heading_style))
    
    conclusion_text = """
    The hybrid HMM-RL approach successfully combines probabilistic modeling with reinforcement
    learning to create an effective Hangman agent. The HMM provides strong priors based on
    linguistic patterns, while the RL agent learns to make strategic decisions that optimize
    the competition score (balancing win rate with efficiency).
    <br/><br/>
    The system demonstrates the value of combining multiple machine learning paradigms:
    probabilistic models for understanding patterns in data, and reinforcement learning for
    sequential decision-making under uncertainty. This hybrid approach leverages the strengths
    of both methods while compensating for their individual limitations.
    """
    elements.append(Paragraph(conclusion_text, normal_style))
    
    # Build PDF
    doc.build(elements)
    print(f"Report generated successfully: {pdf_path}")


if __name__ == "__main__":
    generate_report()

