"""
NVIDIA Stock Price Predictor
----------------------------
Started this as a way to learn sklearn and yfinance properly.
Predicts next-day closing price using engineered features + regression models.

Notes from building this:
- Linear regression alone was surprisingly decent, Ridge helped with overfitting
- Tried LSTM at first but it was overkill for this scope, kept it to regression
- 3 year window felt right — long enough to capture cycles, not too stale
- Predicting raw price vs returns: raw price looks better on charts but returns
  would be more honest. Keeping raw price for now since it's easier to interpret.

TODO: try predicting returns instead of raw price
TODO: add RSI and MACD as features
TODO: experiment with longer lag windows (20, 30 days)
TODO: add a simple backtesting loop to simulate trades
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")


# 3 years gives enough data to capture multiple market cycles
# tried 5 years but NVDA price scale was so different pre-2020 it hurt the model
TICKER = "NVDA"
LOOKBACK = "3y"
TRAIN_RATIO = 0.8  # 80/20 split


def fetch_data(ticker, period):
    t = yf.Ticker(ticker)
    df = t.history(period=period)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    return df


def add_features(df):
    df = df.copy()

    # lag features — yesterday's price is the strongest predictor
    # tried up to lag_20 but anything past 10 didn't add much
    for lag in [1, 2, 3, 5, 10]:
        df[f"lag_{lag}"] = df["Close"].shift(lag)

    # moving averages to capture trend direction
    df["ma_5"]  = df["Close"].rolling(5).mean()
    df["ma_10"] = df["Close"].rolling(10).mean()
    df["ma_20"] = df["Close"].rolling(20).mean()

    # rolling std as a proxy for recent volatility
    df["std_5"]  = df["Close"].rolling(5).std()
    df["std_10"] = df["Close"].rolling(10).std()

    # momentum — how much has price moved over N days
    df["momentum_5"]  = df["Close"] - df["Close"].shift(5)
    df["momentum_10"] = df["Close"] - df["Close"].shift(10)

    # intraday range and open-to-close move
    df["hl_range"]  = df["High"] - df["Low"]
    df["oc_change"] = df["Close"] - df["Open"]

    # volume ratio vs recent average — spikes can signal moves
    df["vol_ma_5"]  = df["Volume"].rolling(5).mean()
    df["vol_ratio"] = df["Volume"] / df["vol_ma_5"]

    # target is next day's close
    df["target"] = df["Close"].shift(-1)

    df.dropna(inplace=True)
    return df


# features that actually moved the needle — dropped raw OHLCV after testing
FEATURES = [
    "lag_1", "lag_2", "lag_3", "lag_5", "lag_10",
    "ma_5", "ma_10", "ma_20",
    "std_5", "std_10",
    "momentum_5", "momentum_10",
    "hl_range", "oc_change",
    "vol_ratio"
]


def build_models():
    # Ridge alpha=10 picked after rough manual tuning, could use CV here
    # Poly degree=2 with high alpha to prevent overfitting from feature explosion
    return {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0))
        ]),
        "Polynomial + Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("model", Ridge(alpha=50.0))  # higher alpha needed with poly features
        ]),
    }


def evaluate(y_true, y_pred):
    return {
        "MAE":  mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2":   r2_score(y_true, y_pred)
    }


def plot_results(dates_test, y_test, predictions, results):
    fig, axes = plt.subplots(3, 1, figsize=(14, 16))
    fig.patch.set_facecolor("#0d1117")

    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e")
        ax.xaxis.label.set_color("#8b949e")
        ax.yaxis.label.set_color("#8b949e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    colors = ["#58a6ff", "#3fb950", "#f78166"]

    # actual vs predicted
    ax = axes[0]
    ax.plot(dates_test, y_test, color="#e6edf3", linewidth=1.5, label="Actual", zorder=3)
    for (name, y_pred), color in zip(predictions.items(), colors):
        ax.plot(dates_test, y_pred, color=color, linewidth=1.2,
                linestyle="--", label=name, alpha=0.85)
    ax.set_title("NVDA — Predicted vs Actual (Test Set)", color="#e6edf3",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Price (USD)")
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # error over time for best model
    best = min(results, key=lambda k: results[k]["MAE"])
    err = predictions[best] - y_test
    ax = axes[1]
    ax.fill_between(dates_test, err, 0, where=(err >= 0),
                    color="#3fb950", alpha=0.5, label="Over-predicted")
    ax.fill_between(dates_test, err, 0, where=(err < 0),
                    color="#f78166", alpha=0.5, label="Under-predicted")
    ax.axhline(0, color="#8b949e", linewidth=0.8, linestyle="--")
    ax.set_title(f"Prediction Error — {best}", color="#e6edf3",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Error (USD)")
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # model comparison bar chart
    ax = axes[2]
    names = list(results.keys())
    maes  = [results[n]["MAE"]  for n in names]
    rmses = [results[n]["RMSE"] for n in names]
    x = np.arange(len(names))
    w = 0.35
    b1 = ax.bar(x - w/2, maes,  w, label="MAE",  color="#58a6ff", alpha=0.85)
    b2 = ax.bar(x + w/2, rmses, w, label="RMSE", color="#3fb950", alpha=0.85)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"${bar.get_height():.1f}", ha="center", va="bottom",
                color="#e6edf3", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, color="#e6edf3", fontsize=9)
    ax.set_title("Model Comparison — MAE vs RMSE", color="#e6edf3",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Error (USD)")
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=9)

    plt.tight_layout(pad=2.5)
    plt.savefig("nvda_prediction_results.png", dpi=150,
                bbox_inches="tight", facecolor="#0d1117")
    print("Chart saved to nvda_prediction_results.png")
    plt.show()


if __name__ == "__main__":
    print(f"Fetching {TICKER} data ({LOOKBACK})...")
    df = fetch_data(TICKER, LOOKBACK)
    print(f"Loaded {len(df)} trading days")

    df = add_features(df)

    X = df[FEATURES].values
    y = df["target"].values
    dates = df.index

    split = int(len(X) * TRAIN_RATIO)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    dates_test = dates[split:]

    print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples\n")

    models = build_models()
    results = {}
    predictions = {}

    print(f"{'Model':<25} {'MAE':>7}  {'RMSE':>8}  {'R2':>7}")
    print("-" * 52)

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        results[name] = evaluate(y_test, y_pred)
        predictions[name] = y_pred
        m = results[name]
        print(f"{name:<25} ${m['MAE']:>6.2f}  ${m['RMSE']:>7.2f}  {m['R2']:>7.4f}")

    print("\nNext-day prediction:")
    last = X[-1].reshape(1, -1)
    for name, pipe in models.items():
        print(f"  {name:<25} ${pipe.predict(last)[0]:.2f}")
    print(f"  {'Last close':<25} ${df['Close'].iloc[-1]:.2f}")

    plot_results(dates_test, y_test, predictions, results)
