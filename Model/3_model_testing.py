"""
PART 3: MODEL TESTING
=====================
- Loads the test split (unseen data — last 10%)
- Runs predictions through RF + XGBoost + Ensemble
- Computes MAE, RMSE, MAPE, R² on test set
- Saves test_results.json and Actual vs Predicted plot
"""

import pandas as pd
import numpy as np
import json
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVED_MODEL_DIR = os.path.join(BASE_DIR, "saved_model")


# ─── Load Everything ──────────────────────────────────────────────────────────
def load_everything():
    with open(os.path.join(SAVED_MODEL_DIR, "config.json")) as f:
        config = json.load(f)

    df = pd.read_parquet(os.path.join(SAVED_MODEL_DIR, "processed_data.parquet"))
    df["Date"] = pd.to_datetime(df["Date"])
    df_clean = df.dropna(subset=config["feature_columns"] + [config["target_column"]])

    n = len(df_clean)
    test = df_clean.iloc[int(n * 0.90):]

    features = config["feature_columns"]
    target   = config["target_column"]

    X_test = test[features].values
    y_test = test[target].values
    dates  = test["Date"].values

    scaler    = joblib.load(os.path.join(SAVED_MODEL_DIR, "scaler.pkl"))
    rf        = joblib.load(os.path.join(SAVED_MODEL_DIR, "rf_model.pkl"))
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(os.path.join(SAVED_MODEL_DIR, "xgb_model.json"))
    meta      = joblib.load(os.path.join(SAVED_MODEL_DIR, "meta_model.pkl"))

    print(f"[3] Test set: {len(y_test)} samples")
    print(f"    Test date range: {pd.Timestamp(dates[0]).date()} → {pd.Timestamp(dates[-1]).date()}")
    return X_test, y_test, dates, scaler, rf, xgb_model, meta, features


# ─── MAPE Helper ──────────────────────────────────────────────────────────────
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9))) * 100


# ─── Run Test Predictions ─────────────────────────────────────────────────────
def run_predictions(X_test, y_test, scaler, rf, xgb_model, meta):
    X_test_s = scaler.transform(X_test)

    rf_preds  = rf.predict(X_test_s)
    xgb_preds = xgb_model.predict(X_test_s)
    X_meta    = np.column_stack([rf_preds, xgb_preds])
    ens_preds = meta.predict(X_meta)

    def metrics_dict(name, preds):
        return {
            f"{name}_mae":  float(mean_absolute_error(y_test, preds)),
            f"{name}_rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
            f"{name}_mape": float(mape(y_test, preds)),
            f"{name}_r2":   float(r2_score(y_test, preds)),
        }

    rf_m  = metrics_dict("rf",  rf_preds)
    xgb_m = metrics_dict("xgb", xgb_preds)
    ens_m = metrics_dict("ens", ens_preds)

    print("[3] Test Metrics:")
    for k, v in {**rf_m, **xgb_m, **ens_m}.items():
        print(f"    {k}: {v:.4f}")

    return rf_preds, xgb_preds, ens_preds, {**rf_m, **xgb_m, **ens_m}


# ─── Save Results ─────────────────────────────────────────────────────────────
def save_results(dates, y_test, rf_preds, xgb_preds, ens_preds, metrics):
    # Build per-day predictions table
    results_rows = []
    for i in range(len(y_test)):
        results_rows.append({
            "date":      str(pd.Timestamp(dates[i]).date()),
            "actual":    float(y_test[i]),
            "rf_pred":   float(rf_preds[i]),
            "xgb_pred":  float(xgb_preds[i]),
            "ens_pred":  float(ens_preds[i]),
        })

    output = {
        "metrics": metrics,
        "predictions": results_rows,
    }
    path = os.path.join(SAVED_MODEL_DIR, "test_results.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[3] Test results saved → {path}")
    return results_rows


# ─── Plot Actual vs Predicted ─────────────────────────────────────────────────
def plot_predictions(dates, y_test, ens_preds):
    if not HAS_PLOT:
        return
    date_vals = [pd.Timestamp(d) for d in dates]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(date_vals, y_test,   label="Actual",    color="#06b6d4", linewidth=1.5)
    ax.plot(date_vals, ens_preds, label="Predicted", color="#a855f7", linewidth=1.5, linestyle="--")
    ax.fill_between(date_vals,
                    np.array(ens_preds) * 0.95,
                    np.array(ens_preds) * 1.05,
                    alpha=0.15, color="#a855f7", label="±5% band")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    ax.set_title("Test Set — Actual vs Predicted Closing Price", fontsize=14, fontweight="bold")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    out = os.path.join(SAVED_MODEL_DIR, "test_actual_vs_predicted.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[3] Plot saved → {out}")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("PART 3: MODEL TESTING — Hold-out Test Set Evaluation")
    print("=" * 60)

    X_test, y_test, dates, scaler, rf, xgb_model, meta, features = load_everything()
    rf_preds, xgb_preds, ens_preds, metrics = run_predictions(X_test, y_test, scaler, rf, xgb_model, meta)
    save_results(dates, y_test, rf_preds, xgb_preds, ens_preds, metrics)
    plot_predictions(dates, y_test, ens_preds)

    print("\n✅ Part 3 complete!")
    print(f"   Ensemble Test → MAE: {metrics['ens_mae']:.4f}, RMSE: {metrics['ens_rmse']:.4f}, "
          f"MAPE: {metrics['ens_mape']:.2f}%, R²: {metrics['ens_r2']:.4f}")
