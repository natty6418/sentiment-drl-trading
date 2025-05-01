# Sentiment-Aware Deep Reinforcement Learning Ensemble for Algorithmic Trading

This repository contains the implementation and evaluation of an ensemble deep reinforcement learning (DRL) pipeline for algorithmic stock trading. The system integrates both sentiment analysis (from Twitter/X and Alpha Vantage news) and technical indicators (RSI, MACD) to improve trading decisions in real-time market environments.

## 🚀 Project Overview

Traders often face difficulties synthesizing sentiment and technical signals effectively, particularly in volatile markets. Our project addresses this challenge by proposing a multimodal DRL architecture that enhances both profitability and risk management.

AY: 2024–2025, NYU Abu Dhabi Engineering Capstone

## 🧠 Architecture

We trained and combined the following DRL agents:

- **TD3 (Twin Delayed Deep Deterministic Policy Gradient)**
- **PPO (Proximal Policy Optimization)**
- **A2C (Advantage Actor-Critic)**

Three ensemble strategies were developed:

1. **Knowledge Distillation:** Averages the actions of experts to train a student network.
2. **Stacked RL:** Uses a meta-agent to select the most appropriate expert.
3. **Mixture of Experts:** Learns to combine expert actions using weighted averages.

These were fused into a unified action vector for live trading decisions.

## 📊 Results

- Ensemble models consistently outperformed individual agents and traditional baselines.
- Enhanced stability with reduced drawdowns and volatility.
- Supports shorting, enabling effectiveness in both bull and bear markets.
- Outperformed benchmarks during market stress scenarios (e.g., 2025 Tarif Meltdown).

## 🧪 Testing and Evaluation

- Portfolio performance comparisons
- Risk-aware metrics (Sharpe Ratio, Drawdown)
- Real-world testing with historical stock data and real-time sentiment feeds

## ⚙️ Constraints

### Technical
- Minimum GPU memory: 16 GB (optimal: 32–64 GB)
- Limited access to high-quality real-time sentiment data

### Non-Technical
- Strict academic timeline for experiments
- Relied on open datasets and models for cost-effectiveness

## 📦 Dependencies

> Will vary based on your implementation, but generally:

- Python 3.8+
- PyTorch / TensorFlow
- OpenAI Gym
- FinRL / Stable Baselines3
- Twitter API / Alpha Vantage API

## 📁 Project Structure

```text
.
├── AlphaVantage.py                 # Script for fetching news sentiment using Alpha Vantage API
├── datasets/
│   ├── dow30_monthly_news_sentiment.csv  # Preprocessed sentiment scores
│   └── merged_df.csv                      # Combined sentiment and stock data
├── fineTunning/
│   ├── finetuned_model/           # Directory for saved fine-tuned models
│   ├── fineTunning.py             # Script to fine-tune language model
│   ├── infer.ipynb                # Inference notebook for sentiment analysis
│   ├── infer.py                   # Script for batch inference
│   ├── requirements.txt           # Dependencies for fine-tuning
│   └── run_finetuning.slurm       # SLURM script for remote training
├── Meta Policy.ipynb              # Notebook for ensemble meta-policy strategy
├── Trading Bot.ipynb              # Main bot logic with evaluation and strategy switching
├── requirements.txt               # Main project dependencies
└── trained_models/
    ├── agent_a2c.zip
    ├── agent_a2c_sentiment.zip
    ├── agent_ddpg.zip
    ├── agent_ddpg_sentiment.zip
    ├── agent_ppo.zip
    ├── agent_ppo_sentiment.zip
    ├── agent_td3.zip
    ├── agent_td3_sentiment.zip
    └── ppo_moe_gating_sb3.zip
```

## 📌 Acknowledgments

Special thanks to our advisors and mentors for their support and guidance:

- Dr. Muhammad Abdullah Hanif  
- Dr. Muhammad Shafique  
- Dr. Pradeep George

## 📜 License

This project is for educational and research purposes. Licensing terms can be added based on your institutional or personal preference.


