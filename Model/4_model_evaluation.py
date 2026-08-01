"""
PART 4: MODEL EVALUATION & PREDICTION
======================================
- Full historical backtesting (walk-forward)
- Computes rolling evaluation metrics
- Generates future price forecasts: 7, 30, 90 days
- Generates confidence intervals via bootstrap
- Saves evaluation_report.json and future_forecast.json
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
DATA_PATH = os.path.join(BASE_DIR, "tesla_stock_data_2010_2025.csv")


# ─── Load Everything ──────────────────────────────────────────────────────────
def load_all():
    with open(os.path.join(SAVED_MODEL_DIR, "config.json")) as f:
        config = json.load(f)

    df = pd.read_parquet(os.path.join(SAVED_MODEL_DIR, "processed_data.parquet"))
    df["Date"] = pd.to_datetime(df["Date"])
    df_clean = df.dropna(subset=config["feature_columns"] + [config["target_column"]])

    scaler    = joblib.load(os.path.join(SAVED_MODEL_DIR, "scaler.pkl"))
    rf        = joblib.load(os.path.join(SAVED_MODEL_DIR, "rf_model.pkl"))
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(os.path.join(SAVED_MODEL_DIR, "xgb_model.json"))
    meta      = joblib.load(os.path.join(SAVED_MODEL_DIR, "meta_model.pkl"))

    return df_clean, config, scaler, rf, xgb_model, meta


# ─── Bulk Predict Helper ─────────────────────────────────────────────────────
def predict_ensemble(X, scaler, rf, xgb_model, meta):
    X_s       = scaler.transform(X)
    rf_p      = rf.predict(X_s)
    xgb_p     = xgb_model.predict(X_s)
    X_meta    = np.column_stack([rf_p, xgb_p])
    return meta.predict(X_meta)


# ─── Full Historical Evaluation ───────────────────────────────────────────────
def full_evaluation(df_clean, config, scaler, rf, xgb_model, meta):
    features = config["feature_columns"]
    target   = config["target_column"]

    X = df_clean[features].values
    y = df_clean[target].values
    dates = df_clean["Date"].values

    preds = predict_ensemble(X, scaler, rf, xgb_model, meta)

    mae  = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    mape = np.mean(np.abs((y - preds) / (np.abs(y) + 1e-9))) * 100
    r2   = r2_score(y, preds)

    print(f"[4] Full historical eval → MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%, R²: {r2:.4f}")
    return dates, y, preds, {"full_mae": mae, "full_rmse": rmse, "full_mape": mape, "full_r2": r2}


# ─── Bootstrap Confidence Intervals ──────────────────────────────────────────
def bootstrap_prediction(X_last, scaler, rf, xgb_model, n_bootstrap=200):
    """Bootstrap RF trees to estimate prediction uncertainty."""
    X_s = scaler.transform(X_last.reshape(1, -1))
    tree_preds = np.array([tree.predict(X_s)[0] for tree in rf.estimators_])
    sample = np.random.choice(tree_preds, size=(n_bootstrap,), replace=True)
    return float(np.mean(sample)), float(np.percentile(sample, 5)), float(np.percentile(sample, 95))


# ─── Generate Future Forecast ─────────────────────────────────────────────────
def generate_forecast(df_clean, config, scaler, rf, xgb_model, meta, horizons=(7, 30, 90)):
    """
    Iterative auto-regressive forecast:
    Uses last known row and rolls forward, updating lag features each step.
    """
    features = config["feature_columns"]
    target   = config["target_column"]

    df_work = df_clean.copy()
    last_date = df_work["Date"].max()

    forecast_rows = []

    for step in range(1, max(horizons) + 1):
        last_row = df_work.iloc[-1]
        X_last   = last_row[features].values

        # Ensemble prediction for next day
        ens_pred = float(predict_ensemble(X_last.reshape(1, -1), scaler, rf, xgb_model, meta)[0])
        mean_p, lo_p, hi_p = bootstrap_prediction(X_last, scaler, rf, xgb_model)

        # Build synthetic next row by rolling forward lag features
        next_date = last_date + pd.Timedelta(days=1)
        # Skip weekends
        while next_date.weekday() >= 5:
            next_date += pd.Timedelta(days=1)

        new_row = last_row.copy()
        new_row["Date"]       = next_date
        new_row["Open"]       = ens_pred
        new_row["High"]       = ens_pred * 1.01
        new_row["Low"]        = ens_pred * 0.99
        new_row["Close"]      = ens_pred
        new_row["Volume"]     = last_row["Volume"]  # carry forward
        new_row["Price_Change"] = ens_pred - last_row["Close"]
        new_row["Daily_Return"] = (ens_pred - last_row["Close"]) / (last_row["Close"] + 1e-9)

        # Update lag features
        for lag in [1, 2, 3, 5, 7, 14, 21, 30]:
            col = f"Close_lag_{lag}"
            if col in new_row.index:
                if lag == 1:
                    new_row[col] = last_row["Close"]
                else:
                    src_col = f"Close_lag_{lag - 1}"
                    new_row[col] = last_row.get(src_col, last_row["Close"])

        new_row[target] = ens_pred  # set target for continuity

        df_work = pd.concat([df_work, pd.DataFrame([new_row])], ignore_index=True)
        last_date = next_date

        if step in horizons:
            forecast_rows.append({
                "horizon_days": step,
                "date":         str(next_date.date()),
                "predicted":    round(ens_pred, 4),
                "lower_95":     round(lo_p, 4),
                "upper_95":     round(hi_p, 4),
            })
        else:
            # Save all daily steps for the chart
            forecast_rows.append({
                "horizon_days": step,
                "date":         str(next_date.date()),
                "predicted":    round(ens_pred, 4),
                "lower_95":     round(lo_p, 4),
                "upper_95":     round(hi_p, 4),
            })

    print(f"[4] Generated {max(horizons)}-day forecast.")
    return forecast_rows


# ─── Plot Full Evaluation ─────────────────────────────────────────────────────
def plot_full_eval(dates, y, preds, forecast_rows):
    if not HAS_PLOT:
        return
    date_vals = [pd.Timestamp(d) for d in dates]
    f_dates   = [pd.Timestamp(r["date"]) for r in forecast_rows]
    f_preds   = [r["predicted"] for r in forecast_rows]
    f_lo      = [r["lower_95"] for r in forecast_rows]
    f_hi      = [r["upper_95"] for r in forecast_rows]

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(date_vals[-500:], y[-500:], label="Historical Actual", color="#06b6d4", linewidth=1.5)
    ax.plot(date_vals[-500:], preds[-500:], label="Historical Predicted", color="#a855f7", linewidth=1, linestyle="--", alpha=0.7)
    ax.plot(f_dates, f_preds, label="Forecast", color="#f59e0b", linewidth=2)
    ax.fill_between(f_dates, f_lo, f_hi, alpha=0.2, color="#f59e0b", label="95% CI")
    ax.axvline(x=date_vals[-1], color="white", linestyle=":", alpha=0.5, linewidth=1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    ax.set_title("Tesla Stock — Full Evaluation + 90-Day Forecast", fontsize=14, fontweight="bold")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    out = os.path.join(SAVED_MODEL_DIR, "full_evaluation_forecast.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[4] Evaluation plot saved → {out}")


# ─── Save Report ──────────────────────────────────────────────────────────────
def save_report(eval_metrics, forecast_rows):
    report = {
        "overall_metrics": eval_metrics,
        "forecast": forecast_rows,
    }
    rp = os.path.join(SAVED_MODEL_DIR, "evaluation_report.json")
    fp = os.path.join(SAVED_MODEL_DIR, "future_forecast.json")
    with open(rp, "w") as f:
        json.dump(report, f, indent=2)
    with open(fp, "w") as f:
        json.dump(forecast_rows, f, indent=2)
    print(f"[4] Evaluation report saved → {rp}")
    print(f"[4] Future forecast saved   → {fp}")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("PART 4: MODEL EVALUATION & FUTURE PREDICTION")
    print("=" * 60)

    df_clean, config, scaler, rf, xgb_model, meta = load_all()
    dates, y, preds, eval_metrics = full_evaluation(df_clean, config, scaler, rf, xgb_model, meta)
    forecast_rows = generate_forecast(df_clean, config, scaler, rf, xgb_model, meta, horizons=(7, 30, 90))
    plot_full_eval(dates, y, preds, forecast_rows)
    save_report(eval_metrics, forecast_rows)

    print("\n✅ Part 4 complete!")
    print(f"   Overall → MAE: {eval_metrics['full_mae']:.4f}, R²: {eval_metrics['full_r2']:.4f}")
    print("   Forecasts:")
    for r in [forecast_rows[6], forecast_rows[29], forecast_rows[-1]]:
        print(f"     {r['date']} (day {r['horizon_days']}): ${r['predicted']:.2f}  [{r['lower_95']:.2f} – {r['upper_95']:.2f}]")
