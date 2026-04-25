"""
NVIDIA Stock Price Predictor
Uses Yahoo Finance + scikit-learn regression models
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
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. FETCH DATA
# ─────────────────────────────────────────────
print("📡 Fetching NVIDIA data from Yahoo Finance...")
ticker = yf.Ticker("NVDA")
df = ticker.history(period="3y")  # 3 years of daily data
df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
df.dropna(inplace=True)
print(f"✅ Loaded {len(df)} trading days\n")


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def add_features(df):
    df = df.copy()

    # Lag features (previous N days' close)
    for lag in [1, 2, 3, 5, 10]:
        df[f"lag_{lag}"] = df["Close"].shift(lag)

    # Rolling statistics
    df["ma_5"]   = df["Close"].rolling(5).mean()
    df["ma_10"]  = df["Close"].rolling(10).mean()
    df["ma_20"]  = df["Close"].rolling(20).mean()
    df["std_5"]  = df["Close"].rolling(5).std()
    df["std_10"] = df["Close"].rolling(10).std()

    # Price momentum
    df["momentum_5"]  = df["Close"] - df["Close"].shift(5)
    df["momentum_10"] = df["Close"] - df["Close"].shift(10)

    # Intraday range
    df["hl_range"]    = df["High"] - df["Low"]
    df["oc_change"]   = df["Close"] - df["Open"]

    # Volume features
    df["vol_ma_5"]    = df["Volume"].rolling(5).mean()
    df["vol_ratio"]   = df["Volume"] / df["vol_ma_5"]

    # Target: next day's closing price
    df["target"] = df["Close"].shift(-1)

    df.dropna(inplace=True)
    return df

df = add_features(df)

feature_cols = [
    "lag_1", "lag_2", "lag_3", "lag_5", "lag_10",
    "ma_5", "ma_10", "ma_20", "std_5", "std_10",
    "momentum_5", "momentum_10",
    "hl_range", "oc_change",
    "vol_ratio"
]
X = df[feature_cols].values
y = df["target"].values
dates = df.index


# ─────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT  (time-aware)
# ─────────────────────────────────────────────
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
dates_test = dates[split:]

print(f"📊 Training samples : {len(X_train)}")
print(f"📊 Testing  samples : {len(X_test)}\n")


# ─────────────────────────────────────────────
# 4. MODELS
# ─────────────────────────────────────────────
models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LinearRegression())
    ]),
    "Ridge Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Ridge(alpha=10.0))
    ]),
    "Polynomial (deg 2) + Ridge": Pipeline([
        ("scaler", StandardScaler()),
        ("poly",   PolynomialFeatures(degree=2, include_bias=False)),
        ("model",  Ridge(alpha=50.0))
    ]),
}

results = {}
predictions = {}

print("=" * 52)
print(f"{'Model':<35} {'MAE':>6}  {'RMSE':>7}  {'R²':>6}")
print("=" * 52)

for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}
    predictions[name] = y_pred

    print(f"{name:<35} ${mae:>5.2f}  ${rmse:>6.2f}  {r2:>6.4f}")

print("=" * 52)


# ─────────────────────────────────────────────
# 5. NEXT-DAY PREDICTION
# ─────────────────────────────────────────────
last_features = X[-1].reshape(1, -1)
print("\n🔮 Next Trading Day Prediction:")
for name, pipe in models.items():
    pred = pipe.predict(last_features)[0]
    print(f"   {name:<35} → ${pred:.2f}")

current_price = df["Close"].iloc[-1]
print(f"\n   Current (last close)               → ${current_price:.2f}")


# ─────────────────────────────────────────────
# 6. PLOTS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 16))
fig.patch.set_facecolor("#0d1117")
for ax in axes:
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#8b949e")
    ax.xaxis.label.set_color("#8b949e")
    ax.yaxis.label.set_color("#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

COLORS = ["#58a6ff", "#3fb950", "#f78166"]

# ── Plot 1: Actual vs Predicted (all models) ──
ax = axes[0]
ax.plot(dates_test, y_test, color="#e6edf3", linewidth=1.5, label="Actual", zorder=3)
for (name, y_pred), color in zip(predictions.items(), COLORS):
    ax.plot(dates_test, y_pred, color=color, linewidth=1.2, linestyle="--",
            label=name, alpha=0.85)
ax.set_title("NVDA — Predicted vs Actual (Test Set)", color="#e6edf3",
             fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Price (USD)", color="#8b949e")
ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

# ── Plot 2: Prediction Error Over Time ──
ax = axes[1]
best_model = min(results, key=lambda k: results[k]["MAE"])
err = predictions[best_model] - y_test
ax.fill_between(dates_test, err, 0,
                where=(err >= 0), color="#3fb950", alpha=0.5, label="Over-predicted")
ax.fill_between(dates_test, err, 0,
                where=(err < 0),  color="#f78166", alpha=0.5, label="Under-predicted")
ax.axhline(0, color="#8b949e", linewidth=0.8, linestyle="--")
ax.set_title(f"Prediction Error — {best_model} (Best Model)", color="#e6edf3",
             fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Error (USD)", color="#8b949e")
ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

# ── Plot 3: Model Comparison Bar Chart ──
ax = axes[2]
model_names = [n.replace(" + ", "\n+\n") for n in results.keys()]
maes  = [v["MAE"]  for v in results.values()]
rmses = [v["RMSE"] for v in results.values()]
x = np.arange(len(model_names))
w = 0.35
bars1 = ax.bar(x - w/2, maes,  w, label="MAE",  color="#58a6ff", alpha=0.85)
bars2 = ax.bar(x + w/2, rmses, w, label="RMSE", color="#3fb950", alpha=0.85)
for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"${bar.get_height():.1f}", ha="center", va="bottom",
            color="#e6edf3", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(model_names, color="#e6edf3", fontsize=9)
ax.set_title("Model Comparison — MAE vs RMSE", color="#e6edf3",
             fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Error (USD)", color="#8b949e")
ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=9)

plt.tight_layout(pad=2.5)
plt.savefig("nvda_prediction_results.png", dpi=150, bbox_inches="tight",
            facecolor="#0d1117")
print("\n📈 Chart saved → nvda_prediction_results.png")
plt.show()
