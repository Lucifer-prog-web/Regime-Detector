📈 Market Regime Detection using Machine Learning & Hidden State Models

An advanced Quantitative Finance + Machine Learning project for detecting market regimes and emotional cycles in financial markets using clustering, probabilistic models, and hidden state sequence modeling.

This project leverages:

K-Means Clustering
Gaussian Mixture Models (GMM)
Hidden Markov Models (HMM)
Hidden Semi-Markov Models (HSMM)
Feature Engineering
Walk-Forward Validation
Regime Transition Analysis
Risk-aware State Detection

The framework is designed to identify latent market conditions such as:

📈 Bull Expansion
⚠️ Transition
📉 Bear Contraction
😰 Panic
🌱 Recovery

and map them to market psychology / emotional cycle states.

🚀 Project Overview

Financial markets transition through hidden behavioral regimes driven by:

Volatility clustering
Investor sentiment
Momentum persistence
Macro uncertainty
Liquidity shifts

Traditional models often fail to capture these latent state transitions.

This project builds a probabilistic market regime detection engine capable of:

Identifying hidden market states
Modeling regime persistence
Detecting transitions dynamically
Mapping emotional market cycles
Improving interpretability of market structure
🧠 Core Mathematical Concepts
1. K-Means Clustering

Used for unsupervised segmentation of market conditions.

Objective:

i=1
∑
k
	​

x∈C
i
	​

∑
	​

∣∣x−μ
i
	​

∣∣
2

Purpose:

Initial market regime discovery
Cluster separation
Pattern grouping
2. Gaussian Mixture Models (GMM)

Probabilistic extension of clustering.

Instead of assigning hard labels, GMM estimates:

P(z∣x)

This enables:

Soft regime assignments
Overlapping market states
Probabilistic uncertainty estimation
3. Hidden Markov Models (HMM)

Models hidden market regimes as latent states.

Transition modeling:

P(S
t
	​

∣S
t−1
	​

)

Used for:

Sequential state inference
Regime persistence detection
Transition probability learning
4. Hidden Semi-Markov Models (HSMM)

Extends HMM by explicitly modeling regime duration.

Advantages:

Better temporal consistency
Reduced noisy state switching
Realistic regime persistence

Critical for modeling:

Extended bull runs
Prolonged bearish sentiment
Slow recovery periods
⚙️ Features Engineered

The model extracts multiple financial indicators:

Market Structure Features
Daily Returns
Rolling Volatility
Drawdown
Price Momentum
Technical Indicators
RSI
Moving Average Spread
Trend Strength
Statistical Features
Rolling Mean
Standard Deviation
Kurtosis
Skewness
Behavioral Indicators
Transition persistence
Regime duration
Volatility bursts
🔄 Regime Transition Framework

The project models transitions across emotional market phases:

Phase	Interpretation
Optimism	Early recovery
Enthusiasm	Growing confidence
Exhilaration	Strong bullish momentum
Euphoria	Peak optimism
Unease	Emerging uncertainty
Denial	Resistance to reversal
Pessimism	Sustained weakness
Despair	Capitulation phase
Hope	Recovery initiation
🧪 Validation Strategy

To avoid overfitting and ensure real-world robustness:

Walk-Forward Testing

The model is trained on rolling historical windows and tested on unseen future data.

This ensures:

Temporal consistency
Out-of-sample robustness
Realistic deployment simulation
📊 Risk Management Layer

Integrated safeguards include:

Volatility targeting
Regime persistence filters
Transition confirmation thresholds
Exposure adjustment logic
Drawdown-aware controls
🛠️ Tech Stack

Programming Language

Python

Libraries

NumPy
Pandas
Scikit-learn
hmmlearn
Matplotlib
Seaborn
SciPy
📂 Project Structure
├── regime_transitions.py     # Regime detection + transition modeling
├── tester.py                 # Validation/testing framework
├── data/                     # Market datasets
├── outputs/                  # Visualizations/results
└── README.md
📈 Key Learnings

Through this project, I gained hands-on experience in:

Unsupervised learning for finance
Sequential probabilistic modeling
Hidden state inference
Regime transition analysis
Quantitative validation frameworks
Financial feature engineering
Research-driven debugging
🔍 Challenges Solved

Key technical challenges addressed:

✔ Regime instability
✔ Overfitting in hidden state models
✔ Noisy transition switching
✔ Optimal state selection
✔ Temporal consistency constraints

🎯 Future Improvements

Planned enhancements:

Bayesian regime inference
Reinforcement learning-based allocation
Dynamic portfolio optimization
Macro-feature integration
Real-time regime dashboard
Explainable AI for state interpretation
🙏 Acknowledgements

Special thanks to Kalp Shah for continuous mentorship, guidance, and support throughout this project.

Your insights and feedback were instrumental in shaping the research process and strengthening the modeling framework.

📬 Connect With Me

If you're interested in:

Quantitative Finance
Market Regime Modeling
Financial Machine Learning
Hidden State Models

Feel free to connect and discuss ideas.

⭐ If you found this project interesting, consider starring the repository.<img width="1465" height="630" alt="legend" src="https://github.com/user-attachments/assets/50d8eca9-847e-40f8-bcde-4eeb379ae60a" />
<img width="1600" height="769" alt="dashboard" src="https://github.com/user-attachments/assets/92e95d6f-519a-414b-9914-38a4569af072" />
<img width="1600" height="768" alt="backtest" src="https://github.com/user-attachments/assets/349d8912-e9bb-46b4-b19d-e72b2f4e640c" />
