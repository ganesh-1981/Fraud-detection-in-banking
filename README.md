# Fraud Detection in Banking Data using Machine Learning Techniques

This project implements advanced machine learning methods to detect fraudulent banking transactions in highly imbalanced datasets. It includes pre-processing, feature selection, ensemble learning, hyperparameter tuning using Bayesian optimization, and deep learning.

## Problem Statement

With the growth of digital banking and credit card usage, fraud detection has become a significant challenge. Fraudulent transactions are rare compared to legitimate ones, leading to imbalanced datasets that require special attention. The aim of this project is to effectively classify fraudulent transactions using optimized machine learning and deep learning techniques.

## Techniques Used

- Bayesian Optimization for hyperparameter tuning  
- LightGBM, XGBoost, and CatBoost for ensemble learning  
- Deep Neural Network with ReLU and Sigmoid activations  
- Precision-Recall and MCC metrics for imbalanced evaluation  
- Feature selection using Information Gain  

## Dataset

- **Source**: [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
- **Size**: 284,807 transactions (492 frauds)  
- **Class Ratio**: 0.172% fraud, 99.82% legitimate  
- **Features**: PCA-transformed features V1–V28, Time, Amount  

## Project Structure
