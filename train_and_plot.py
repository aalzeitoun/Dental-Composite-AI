import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json

from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
import shap


# This script will load the processed data, train the models, 
# compute metrics and SHAP values, and generate all the plots.

# ----------------------------
# Config & Load Data
# ----------------------------
PROCESSED_DATA_FILE = "processed_data.joblib"
FI_LGBM_CSV = "feature_importance_lightgbm.csv"
FI_CAT_CSV = "feature_importance_catboost.csv"
SHAP_CSV = "shap_mean_abs_lightgbm.csv"
METRICS_JSON = "metrics.json"

print(f"Loading data from {PROCESSED_DATA_FILE}...")
data = joblib.load(PROCESSED_DATA_FILE)
X = data['X']
y = data['y']
groups = data['groups']
feature_names = data['feature_names']
target = data['target']
SEED = data['SEED']

cv = GroupKFold(n_splits=5)

# ----------------------------
# Train LightGBM (CV)
# ----------------------------
print("Training LightGBM model...")
lgb_r2s, lgb_mae, lgb_rmse = [], [], []
lgb_importance = np.zeros(len(feature_names), dtype=float)

for tr, te in cv.split(X, y, groups):
    model = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.05, num_leaves=31, random_state=SEED
    )
    model.fit(X[tr], y[tr])
    pred = model.predict(X[te])
    lgb_r2s.append(r2_score(y[te], pred))
    lgb_mae.append(mean_absolute_error(y[te], pred))
    lgb_rmse.append(np.sqrt(mean_squared_error(y[te], pred)))
    lgb_importance += model.feature_importances_

lgb_importance /= cv.get_n_splits()
fi_lgb = pd.DataFrame({"feature": feature_names, "importance": lgb_importance}) \
            .sort_values("importance", ascending=False)
fi_lgb.to_csv(FI_LGBM_CSV, index=False)
print(f"Saved LightGBM feature importance -> {FI_LGBM_CSV}")


# ----------------------------
# Train CatBoost (CV)
# ----------------------------
print("Training CatBoost model...")
cat_r2s, cat_mae, cat_rmse = [], [], []
cat_importance = np.zeros(len(feature_names), dtype=float)

for tr, te in cv.split(X, y, groups):
    train_pool = Pool(X[tr], y[tr], feature_names=feature_names)
    test_pool = Pool(X[te], y[te], feature_names=feature_names)
    model = CatBoostRegressor(
        iterations=600, learning_rate=0.05, depth=6, random_seed=SEED, verbose=0
    )
    model.fit(train_pool)
    pred = model.predict(test_pool)
    cat_r2s.append(r2_score(y[te], pred))
    cat_mae.append(mean_absolute_error(y[te], pred))
    cat_rmse.append(np.sqrt(mean_squared_error(y[te], pred)))
    cat_importance += np.array(model.get_feature_importance(train_pool))

cat_importance /= cv.get_n_splits()
fi_cat = pd.DataFrame({"feature": feature_names, "importance": cat_importance}) \
            .sort_values("importance", ascending=False)
fi_cat.to_csv(FI_CAT_CSV, index=False)
print(f"Saved CatBoost feature importance -> {FI_CAT_CSV}")


# ----------------------------
# SHAP Global Importance (LightGBM fit on all data)
# ----------------------------
print("Computing SHAP values...")
lgbm_full = lgb.LGBMRegressor(
    n_estimators=600, learning_rate=0.05, num_leaves=31, random_state=SEED
)
lgbm_full.fit(X, y)
explainer = shap.TreeExplainer(lgbm_full)
shap_values = explainer.shap_values(X)
mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_df = pd.DataFrame(
    {"feature": feature_names, "mean_abs_shap": mean_abs_shap}
).sort_values("mean_abs_shap", ascending=False)
shap_df.to_csv(SHAP_CSV, index=False)
print(f"Saved SHAP global importance -> {SHAP_CSV}")


# ----------------------------
# Save metrics
# ----------------------------
metrics = {
    "LightGBM": {
        "R2_mean": float(np.mean(lgb_r2s)),
        "R2_std": float(np.std(lgb_r2s)),
        "MAE_mean": float(np.mean(lgb_mae)),
        "RMSE_mean": float(np.mean(lgb_rmse)),
    },
    "CatBoost": {
        "R2_mean": float(np.mean(cat_r2s)),
        "R2_std": float(np.std(cat_r2s)),
        "MAE_mean": float(np.mean(cat_mae)),
        "RMSE_mean": float(np.mean(cat_rmse)),
    },
    "files": {
        "fi_lightgbm_csv": FI_LGBM_CSV,
        "fi_catboost_csv": FI_CAT_CSV,
        "shap_lightgbm_csv": SHAP_CSV,
    },
}

with open(METRICS_JSON, "w") as f:
    json.dump(metrics, f, indent=2)

print("\n=== Cross-validated Results ===")
for k, v in metrics.items():
    if k == "files":
        continue
    print(
        f"{k}: R2={v['R2_mean']:.3f}±{v['R2_std']:.3f} | "
        f"MAE={v['MAE_mean']:.2f} | RMSE={v['RMSE_mean']:.2f}"
    )
print("\nArtifacts saved:", metrics["files"])


# ----------------------------
# PLOTS
# ----------------------------

print("\n=== Generating Plots ===")

# --- Plot 1: Feature Importance Bar Chart ---
try:
    lgbm_fi_df = pd.read_csv(FI_LGBM_CSV)
    lgbm_fi_df = lgbm_fi_df.sort_values(by="importance", ascending=True)
    plt.figure(figsize=(10, 8))
    plt.barh(lgbm_fi_df["feature"], lgbm_fi_df["importance"], color='skyblue')
    plt.title("LightGBM Feature Importance", fontsize=16)
    plt.xlabel("Importance", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.tight_layout()
    plt.show()
except FileNotFoundError:
    print(f"Error: '{FI_LGBM_CSV}' not found. Cannot generate plot.")

# --- Plot 2: SHAP Global Importance Bar Chart ---
try:
    shap_df = pd.read_csv(SHAP_CSV)
    shap_df = shap_df.sort_values(by="mean_abs_shap", ascending=True)
    plt.figure(figsize=(10, 8))
    plt.barh(shap_df["feature"], shap_df["mean_abs_shap"], color='lightcoral')
    plt.title("LightGBM SHAP Global Importance (Mean Absolute SHAP Value)", fontsize=16)
    plt.xlabel("Mean Absolute SHAP Value", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.tight_layout()
    plt.show()
except FileNotFoundError:
    print(f"Error: '{SHAP_CSV}' not found. Cannot generate plot.")

# --- Plot 3: Actual vs. Predicted Scatter Plot ---
all_y_true = []
all_y_pred = []
all_residuals = []
for tr, te in cv.split(X, y, groups):
    model = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=31, random_state=SEED)
    model.fit(X[tr], y[tr])
    pred = model.predict(X[te])
    all_y_true.extend(y[te])
    all_y_pred.extend(pred)
    all_residuals.extend(y[te] - pred)

plt.figure(figsize=(8, 8))
plt.scatter(all_y_true, all_y_pred, alpha=0.6, color='darkblue')
min_val = min(min(all_y_true), min(all_y_pred))
max_val = max(max(all_y_true), max(all_y_pred))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
plt.title("Actual vs. Predicted Values (LightGBM)", fontsize=16)
plt.xlabel("Actual Flexural Strength (MPa)", fontsize=12)
plt.ylabel("Predicted Flexural Strength (MPa)", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot 4: Residuals Plot ---
plt.figure(figsize=(10, 6))
plt.scatter(all_y_pred, all_residuals, alpha=0.6, color='darkorange')
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.title("Residuals Plot", fontsize=16)
plt.xlabel("Predicted Flexural Strength (MPa)", fontsize=12)
plt.ylabel("Residuals (Actual - Predicted)", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot 5: Cross-Validation Fold Comparison ---
metrics_df = pd.DataFrame({
    'R2': lgb_r2s,
    'MAE': lgb_mae,
    'RMSE': lgb_rmse
})
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
metrics_df['R2'].plot(kind='bar', color='darkgreen')
plt.title('R² per Fold', fontsize=14)
plt.ylabel('R² Score', fontsize=12)
plt.xlabel('Fold', fontsize=12)
plt.ylim(0, 1.05)
plt.subplot(1, 3, 2)
metrics_df['MAE'].plot(kind='bar', color='darkblue')
plt.title('MAE per Fold', fontsize=14)
plt.ylabel('MAE (MPa)', fontsize=12)
plt.xlabel('Fold', fontsize=12)
plt.subplot(1, 3, 3)
metrics_df['RMSE'].plot(kind='bar', color='darkred')
plt.title('RMSE per Fold', fontsize=14)
plt.ylabel('RMSE (MPa)', fontsize=12)
plt.xlabel('Fold', fontsize=12)
plt.tight_layout()
plt.show()

# --- Plot 6: SHAP Dependency Plot ---
feature_to_plot = 'filler_wt'
feature_idx = feature_names.index(feature_to_plot)
shap.dependence_plot(
    ind=feature_idx,
    shap_values=shap_values,
    features=X,
    feature_names=feature_names,
    display_features=pd.DataFrame(X, columns=feature_names)
)
plt.title(f"SHAP Dependency Plot for {feature_to_plot}", fontsize=16)
plt.show()