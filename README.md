# Bike Sharing Demand Forecasting: A Hybrid Residual Learning Approach

## Project Description
This project focuses on predicting hourly bike rental demand using the Kaggle Bike Sharing Demand dataset. The core innovation lies in a hybrid forecasting architecture that disentangles global time-series trends from local feature-driven variations. By combining statistical forecasting with machine learning ensembles, the system effectively handles complex seasonality and high-variance demand patterns.

## Key Achievements
* **Final Performance:** Achieved a top-tier RMSLE of **$0.37849$** on the Kaggle Leaderboard.
* **Architecture:** Developed a robust **Residual Learning** pipeline.
* **Engineering:** Implemented professional modular Python code with structured logging, advanced validation, and ensemble strategies.

---

## Core Methodology

### 1. Hybrid Architecture (Prophet + Tree Ensemble)
The model follows a two-stage **Residual Learning** strategy to optimize prediction accuracy:
* **Global Component (Prophet):** Handles long-term trends, multiple seasonalities (daily, weekly, yearly), and holiday effects in log-space.
* **Local Component (Trees):** An ensemble of **XGBoost, LightGBM, and CatBoost** models trained specifically to predict the residuals (errors) of the Prophet model. This allows the trees to focus solely on short-term factors like weather fluctuations and hourly peak-hour dynamics.

### 2. Advanced Engineering Techniques
* **Expanding Window Validation:** Utilized an **ExpandingWindowSplitter** to maintain chronological order and prevent data leakage, ensuring a realistic performance estimate.
* **Specialized Preprocessing:** Implemented algorithmic-specific pipelines: **Ordinal Encoding** for CatBoost to leverage its internal categorical handling, and **One-Hot Encoding** for XGBoost/LightGBM.
* **Early Stopping:** Integrated an internal 10% watchlist validation for each fold to dynamically stop training, preventing overfitting while allowing high `n_estimators`.

### 3. Inference & Post-Processing
* **True Seed Averaging:** The final inference engine averages predictions across **9 models** (3 random seeds per model type) to minimize variance and increase generalization.
* **Weighted Blending:** Applied a strategic blend ($50\%$ CatBoost, $25\%$ XGBoost, $25\%$ LightGBM) optimized through rigorous cross-validation.
* **Rolling Median Smoothing:** Applied a 3-hour window rolling median to the final predictions to neutralize noisy spikes and outliers characteristic of tree-based models.

---

## Tech Stack
- **Language:** Python  
- **Time-series:** Prophet  
- **Algorithms:** CatBoost, XGBoost, LightGBM  
- **Data / ML:** Pandas, NumPy, Scikit-learn  
- **Tuning:** Optuna

---

## Project Structure
```text
├── src/
│   ├── config.py           # Global settings, feature lists, and hyperparameters
│   ├── dataloader.py       # Modular data loading and cleaning
│   ├── features_new.py     # Custom feature transformers (Weather Lags, Day/Night)
│   ├── pipeline.py         # Specialized preprocessing and model factory
│   ├── utils.py            # Training loops, Prophet logic, and inference engines
│   └── train.py            # Main execution script for the training pipeline
├── blend_submission.py     # Final ensemble blending and smoothing script
└── README.md               # Project documentation
