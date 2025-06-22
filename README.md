# Machine Learning Traditional Projects

This repository contains two end-to-end machine learning projects focused on real-world applications — one on structured classification and the other on time series forecasting. Each project includes detailed data processing, modeling techniques, and evaluation strategies for practical deployment.

---

## 1. Essential ML Techniques in Classification: A Loan Approval Case Study

**Use Case**  
Build a binary classifier to predict whether a loan should be approved based on applicant data (e.g., income, marital status, loan amount). The project goes beyond a simple baseline and dives into data preprocessing, handling class imbalance, tuning models, and addressing business-critical misclassification costs.

**Tech Stack**  
- **Dataset**: Structured loan application data  
- **Preprocessing**:  
  - Missing value imputation, outlier handling  
  - Label encoding, feature engineering, skewness reduction, scaling  
- **Exploratory Analysis**: Categorical/numerical feature analysis, correlation matrix  
- **Modeling**:  
  - Baseline with `DecisionTreeClassifier`  
  - Experiments with `RandomForestClassifier`, `XGBoost`, `MLPClassifier`, `LogisticRegression`  
  - Hyperparameter tuning with GridSearchCV  
- **Evaluation**:  
  - F1-score, confusion matrix, precision-recall tradeoffs  
  - Misclassification cost analysis  
  - Strategies for imbalanced data (e.g., class weights, SMOTE)

---

## 2. Stock Price Forecasting with LSTM

**Use Case**  
Train an `LSTM` model to predict the 21st day’s stock price using the past 20 days of adjusted close prices. This project explores deep learning for time series and highlights both the promise and limitations of stock prediction.

**Tech Stack**  
- **Data Source**: Yahoo Finance API (Adjusted Close Prices)  
- **Preprocessing**:  
  - Compute 20-day & 50-day moving averages, RSI  
  - Normalize using `MinMaxScaler`  
  - Supervised dataset: past 20-day window → next-day price  
- **Modeling**:  
  - LSTM layer with dropout for regularization  
  - Walk-forward cross-validation (rolling window)  
  - Time-aware train/validation splits  
- **Observations**:  
  - Model tends to return a shifted version of input  
  - Reflects the noisy, stochastic nature of stock prices  
  - Useful lesson in understanding limits of LSTM on financial data

---

🔍 These projects emphasize practical ML workflows, including data cleaning, feature design, model tuning, and thoughtful evaluation. They're ideal for anyone looking to go deeper than notebooks and into production-quality work.
