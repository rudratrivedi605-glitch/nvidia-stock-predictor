# NVIDIA Stock Price Predictor

A machine learning project that predicts NVIDIA's next-day closing stock price using regression models and historical data pulled from Yahoo Finance.

## Overview

This project fetches 3 years of NVIDIA (NVDA) daily stock data, engineers features from the raw price data, trains three regression models, and evaluates their performance against real prices. It also predicts the next trading day's closing price.

## Models Used

- Linear Regression
- Ridge Regression
- Polynomial Ridge Regression (degree 2)

## Features Engineered

- Lag prices from 1, 2, 3, 5, and 10 days back
- Moving averages over 5, 10, and 20 days
- Price momentum over 5 and 10 days
- Rolling standard deviation
- Intraday high/low range
- Open to close price change
- Volume ratio

## Evaluation Metrics

- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- R2 Score

## Setup

Install dependencies:

pip install yfinance pandas scikit-learn matplotlib

## Run

python3 nvidia_predictor.py

## Output

- Metrics table comparing all three models printed in terminal
- Next-day closing price prediction
- Chart saved as nvda_prediction_results.png showing predicted vs actual prices, prediction error over time, and a model comparison bar chart

## Tech Stack

- Python
- yfinance
- pandas
- scikit-learn
- matplotlib

## Notes

Regression models perform well at following general price trends but struggle with sudden large price movements. This project is intended for learning purposes and should not be used for real trading decisions.
