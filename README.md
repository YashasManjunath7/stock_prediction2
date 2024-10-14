Hybrid LSTM-CNN-GRU Model for Stock Price Prediction
Project Overview
This project implements a hybrid machine learning model for stock price prediction, specifically designed for predicting standardized daily returns of Apple Inc. (AAPL) stock. The model combines the strengths of three popular neural network architectures:

Convolutional Neural Networks (CNN): For feature extraction from time series data.
Long Short-Term Memory (LSTM): For capturing long-term dependencies in sequential data.
Gated Recurrent Units (GRU): For efficient processing of sequential data with fewer computational resources.
The architecture is further enhanced by using TimeDistributed LSTM layers and LeakyReLU activation functions. The project includes comprehensive performance metrics, data preprocessing, model training, and visualization of results.

Key Features
Hybrid Model: Combines CNN, LSTM, and GRU layers for superior sequential processing and feature extraction.
LeakyReLU Activation: Mitigates the "dying ReLU" problem, ensuring smoother learning.
TimeDistributed Layer: Applies a Dense layer to each time step of the LSTM outputs.
Performance Metrics: Includes RMSE, MAE, R² (Coefficient of Determination), and Directional Accuracy.
Data Visualization: Plots actual vs. predicted returns and model training progress.
Technologies Used
Python 3.x
TensorFlow / Keras
Yahoo Finance API (via yfinance)
Matplotlib (for visualizations)
Pandas / NumPy (for data manipulation and preprocessing)
Dataset
The stock data is sourced directly from Yahoo Finance using the yfinance library. The dataset includes historical data of Apple Inc. (AAPL) stock from January 1, 2014, to January 1, 2024. The following features are used:

Close Price: Daily closing price of the stock.
Return: Daily percentage return, calculated from the closing prices.
Simple Moving Average (SMA): 5-day simple moving average.
Exponential Moving Average (EMA): 5-day exponential moving average.
Model Architecture
The hybrid model consists of the following layers:

Convolutional Layer: Extracts features from the input sequences.
MaxPooling Layer: Reduces dimensionality and filters significant features.
LSTM Layer: Captures long-term temporal dependencies.
TimeDistributed Dense Layer: Applies a Dense layer to each time step's output from the LSTM.
GRU Layer: Efficiently processes sequential data with fewer parameters.
Dense + LeakyReLU Layers: Final prediction layers.
The model is compiled using the Adam optimizer and trained on mean squared error (MSE) loss.
