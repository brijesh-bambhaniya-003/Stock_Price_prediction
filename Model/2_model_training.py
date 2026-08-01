"""
PART 2: MODEL TRAINING
======================
- Loads the processed dataset from Part 1
- Trains a Random Forest Regressor and an XGBoost Regressor
- Stacks them with a Ridge meta-learner (ensemble)
- Saves trained models + scaler to saved_model/
- Logs and plots training metrics
"""

import pandas as pd
import numpy as np
import json
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVED_MODEL_DIR = os.path.join(BASE_DIR, "saved_model")


# ─── Load Config & Data ───────────────────────────────────────────────────────
def load_config_and_data():
    with open(os.path.join(SAVED_MODEL_DIR, "config.json")) as f:
        config = json.load(f)

    df = pd.read_parquet(os.path.join(SAVED_MODEL_DIR, "processed_data.parquet"))
    df["Date"] = pd.to_datetime(df["Date"])
    df_clean = df.dropna(subset=config["feature_columns"] + [config["target_column"]])

    n = len(df_clean)
    train_end = int(n * 0.80)
    val_end   = int(n * 0.90)

    train = df_clean.iloc[:train_end]
    val   = df_clean.iloc[train_end:val_end]

    features = config["feature_columns"]
    target   = config["target_column"]

    X_train = train[features].values
    y_train = train[target].values
    X_val   = val[features].values
    y_val   = val[target].values

    print(f"[2] Train: {X_train.shape}, Val: {X_val.shape}")
    return X_train, y_train, X_val, y_val, features, config


# ─── Scale Features ───────────────────────────────────────────────────────────
def scale_features(X_train, X_val):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    joblib.dump(scaler, os.path.join(SAVED_MODEL_DIR, "scaler.pkl"))
    print("[2] Scaler saved.")
    return X_train_s, X_val_s, scaler


# ─── Train Random Forest ──────────────────────────────────────────────────────
def train_random_forest(X_train, y_train, X_val, y_val):
    print("[2] Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)
    preds = rf.predict(X_val)
    mae  = mean_absolute_error(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    r2   = r2_score(y_val, preds)
    print(f"   RF  → MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    joblib.dump(rf, os.path.join(SAVED_MODEL_DIR, "rf_model.pkl"))
    return rf, preds, {"rf_mae": mae, "rf_rmse": rmse, "rf_r2": r2}


# ─── Train XGBoost ────────────────────────────────────────────────────────────
def train_xgboost(X_train, y_train, X_val, y_val):
    print("[2] Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    preds = xgb_model.predict(X_val)
    mae  = mean_absolute_error(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    r2   = r2_score(y_val, preds)
    print(f"   XGB → MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    xgb_model.save_model(os.path.join(SAVED_MODEL_DIR, "xgb_model.json"))
    return xgb_model, preds, {"xgb_mae": mae, "xgb_rmse": rmse, "xgb_r2": r2}


# ─── Train Ensemble (Stacking Meta-Learner) ───────────────────────────────────
def train_ensemble(rf_preds, xgb_preds, y_val):
    print("[2] Training Ensemble meta-learner (Ridge)...")
    X_meta = np.column_stack([rf_preds, xgb_preds])
    meta = Ridge(alpha=1.0)
    meta.fit(X_meta, y_val)
    ensemble_preds = meta.predict(X_meta)
    mae  = mean_absolute_error(y_val, ensemble_preds)
    rmse = np.sqrt(mean_squared_error(y_val, ensemble_preds))
    r2   = r2_score(y_val, ensemble_preds)
    print(f"   ENS → MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    joblib.dump(meta, os.path.join(SAVED_MODEL_DIR, "meta_model.pkl"))
    return meta, ensemble_preds, {"ens_mae": mae, "ens_rmse": rmse, "ens_r2": r2}


# ─── Plot Feature Importance ──────────────────────────────────────────────────
def plot_feature_importance(rf, features, top_n=20):
    if not HAS_PLOT:
        return
    importances = rf.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(top_n), importances[indices], color="#7c3aed")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([features[i] for i in indices], fontsize=9)
    ax.set_title("Top Feature Importances (Random Forest)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    out = os.path.join(SAVED_MODEL_DIR, "feature_importance.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[2] Feature importance plot saved → {out}")


# ─── Save Training Metrics ────────────────────────────────────────────────────
def save_training_metrics(rf_m, xgb_m, ens_m):
    metrics = {**rf_m, **xgb_m, **ens_m}
    path = os.path.join(SAVED_MODEL_DIR, "training_metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[2] Training metrics saved → {path}")
    return metrics


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("PART 2: MODEL TRAINING — Random Forest + XGBoost Ensemble")
    print("=" * 60)

    X_train, y_train, X_val, y_val, features, config = load_config_and_data()
    X_train_s, X_val_s, scaler = scale_features(X_train, X_val)

    rf, rf_preds, rf_m   = train_random_forest(X_train_s, y_train, X_val_s, y_val)
    xgb_m_obj, xgb_preds, xgb_m = train_xgboost(X_train_s, y_train, X_val_s, y_val)
    meta, ens_preds, ens_m = train_ensemble(rf_preds, xgb_preds, y_val)

    plot_feature_importance(rf, features)
    metrics = save_training_metrics(rf_m, xgb_m, ens_m)

    print("\n✅ Part 2 complete! Models saved to saved_model/")
    print(f"   Ensemble Val → MAE: {ens_m['ens_mae']:.4f}, RMSE: {ens_m['ens_rmse']:.4f}, R²: {ens_m['ens_r2']:.4f}")
