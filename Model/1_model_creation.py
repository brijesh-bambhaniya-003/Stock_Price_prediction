"""
PART 1: MODEL CREATION
======================
- Loads the Tesla stock CSV dataset
- Engineers features: lag prices, rolling stats, RSI, MACD, technical indicators
- Defines train/validation/test splits (time-based: 80/10/10)
- Defines model architecture: Random Forest + XGBoost ensemble
- Saves the feature list and config for downstream scripts
"""

import pandas as pd
import numpy as np
import json
import os
import sys

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "tesla_stock_data_2010_2025.csv")
SAVED_MODEL_DIR = os.path.join(BASE_DIR, "saved_model")
os.makedirs(SAVED_MODEL_DIR, exist_ok=True)


# ─── 1. Load Data ─────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"[1] Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"    Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
    return df


# ─── 2. Feature Engineering ───────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Lag features (previous closes)
    for lag in [1, 2, 3, 5, 7, 14, 21, 30]:
        df[f"Close_lag_{lag}"] = df["Close"].shift(lag)

    # Rolling statistics
    for window in [7, 14, 30, 60, 90]:
        df[f"roll_mean_{window}"] = df["Close"].shift(1).rolling(window).mean()
        df[f"roll_std_{window}"] = df["Close"].shift(1).rolling(window).std()
        df[f"roll_min_{window}"] = df["Close"].shift(1).rolling(window).min()
        df[f"roll_max_{window}"] = df["Close"].shift(1).rolling(window).max()

    # RSI (14 period)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # Bollinger Bands
    roll20 = df["Close"].shift(1).rolling(20)
    df["BB_upper"] = roll20.mean() + 2 * roll20.std()
    df["BB_lower"] = roll20.mean() - 2 * roll20.std()
    df["BB_width"] = df["BB_upper"] - df["BB_lower"]

    # Price-derived features
    df["HL_ratio"] = df["High"] / (df["Low"] + 1e-9)
    df["OC_ratio"] = df["Open"] / (df["Close"] + 1e-9)
    df["log_volume"] = np.log1p(df["Volume"])

    # Encode Day_of_Week (may be string like "Monday") → integer 0-6
    dow_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
               "Friday": 4, "Saturday": 5, "Sunday": 6}
    if df["Day_of_Week"].dtype == object:
        df["Day_of_Week"] = df["Day_of_Week"].map(dow_map).fillna(df["Date"].dt.dayofweek)
    df["Day_of_Week"] = df["Day_of_Week"].astype(float)

    # Derive numeric date features from Date column (reliable)
    df["Month"]   = df["Date"].dt.month.astype(float)
    df["Quarter"] = df["Date"].dt.quarter.astype(float)

    # Target: next day's close price
    df["Target"] = df["Close"].shift(-1)

    print(f"[1] Feature engineering done. Total columns: {df.shape[1]}")
    return df


# ─── 3. Define Features & Target ─────────────────────────────────────────────
FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "Daily_Return", "Price_Range", "MA_7", "MA_30", "MA_90",
    "Volatility_7d", "Month", "Day_of_Week", "Quarter",
    "RSI_14", "MACD", "MACD_signal", "MACD_hist",
    "BB_upper", "BB_lower", "BB_width",
    "HL_ratio", "OC_ratio", "log_volume",
] + [f"Close_lag_{l}" for l in [1, 2, 3, 5, 7, 14, 21, 30]] \
  + [f"roll_mean_{w}" for w in [7, 14, 30, 60, 90]] \
  + [f"roll_std_{w}" for w in [7, 14, 30, 60, 90]] \
  + [f"roll_min_{w}" for w in [7, 14, 30, 60, 90]] \
  + [f"roll_max_{w}" for w in [7, 14, 30, 60, 90]]

TARGET_COLUMN = "Target"


# ─── 4. Train / Val / Test Split (time-based) ────────────────────────────────
def split_data(df: pd.DataFrame):
    df_clean = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()
    n = len(df_clean)
    train_end = int(n * 0.80)
    val_end   = int(n * 0.90)

    train = df_clean.iloc[:train_end]
    val   = df_clean.iloc[train_end:val_end]
    test  = df_clean.iloc[val_end:]

    print(f"[1] Split → Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test


# ─── 5. Save Config ───────────────────────────────────────────────────────────
def save_config(train, val, test):
    config = {
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "train_date_range": [str(train["Date"].min().date()), str(train["Date"].max().date())],
        "val_date_range":   [str(val["Date"].min().date()),   str(val["Date"].max().date())],
        "test_date_range":  [str(test["Date"].min().date()),  str(test["Date"].max().date())],
    }
    config_path = os.path.join(SAVED_MODEL_DIR, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[1] Config saved → {config_path}")
    return config


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("PART 1: MODEL CREATION — Feature Engineering & Data Split")
    print("=" * 60)

    df = load_data(DATA_PATH)
    df = engineer_features(df)
    train, val, test = split_data(df)
    config = save_config(train, val, test)

    # Save processed dataset for downstream scripts (parquet + CSV fallback)
    processed_path = os.path.join(SAVED_MODEL_DIR, "processed_data.parquet")
    csv_path       = os.path.join(SAVED_MODEL_DIR, "processed_data.csv")
    try:
        df.to_parquet(processed_path, index=False)
        print(f"[1] Processed data (parquet) saved → {processed_path}")
    except Exception:
        print("[1] pyarrow not available, skipping parquet.")
    df.to_csv(csv_path, index=False)
    print(f"[1] Processed data (CSV)     saved → {csv_path}")

    print("\n✅ Part 1 complete! Config and processed data saved to saved_model/")
    print(f"   Features: {len(FEATURE_COLUMNS)}")
    print(f"   Config: {json.dumps(config, indent=2)}")
