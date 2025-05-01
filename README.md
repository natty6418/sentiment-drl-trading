# Sentiment-Aware Deep Reinforcement Learning Ensemble for Algorithmic Trading

This repository contains the implementation and evaluation of an ensemble deep reinforcement learning (DRL) pipeline for algorithmic stock trading. The system integrates both sentiment analysis (from Twitter/X and Alpha Vantage news) and technical indicators (RSI, MACD) to improve trading decisions in real-time market environments.

## 🚀 Project Overview

Traders often face difficulties synthesizing sentiment and technical signals effectively, particularly in volatile markets. Our project addresses this challenge by proposing a multimodal DRL architecture that enhances both profitability and risk management.

## 👨‍💻 Authors

- Walid Al-Eisawi  
- Natty Metekie  
- Abay Oralov  
- Hamdan Zoghbor  

Capstone Coordinators:  
- Pradeep George  
- Muhammad Abdullah Hanif  
- Muhammad Shafique  

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

├── data/ # Preprocessed financial & sentiment datasets ├── agents/ # TD3, PPO, A2C implementations ├── ensemble/ # Distillation, Stacked RL, Mixture of Experts ├── evaluation/ # Performance metrics & visualizations ├── notebooks/ # Experiment tracking & result plots ├── README.md


## 📌 Acknowledgments

Special thanks to our advisors and mentors for their support and guidance:

- Dr. Muhammad Abdullah Hanif  
- Dr. Muhammad Shafique  
- Dr. Pradeep George

## 📜 License

This project is for educational and research purposes. Licensing terms can be added based on your institutional or personal preference.


