import os, json, re
import numpy as np
import pandas as pd
import joblib

# This script will handle all the data loading, cleaning, feature engineering, and augmentation. 
# It will save the processed data as a .joblib file, which is an efficient way to store Python objects.

# ----------------------------
# Config
# ----------------------------
SEED = 42
N_AUG = 9
INPUT_CSV = "original_dataset.xlsx"
AUG_CSV = "augmented_features.csv"
PROCESSED_DATA_FILE = "processed_data.joblib"

np.random.seed(SEED)

# ----------------------------
# Define Target Variable
# ----------------------------
target = "flexural_strength_mpa"

# ----------------------------
# Utilities
# ----------------------------
def parse_filler_percent(val):
    """Extract numeric 'wt' and 'vol' from text like '77 wt%, 60 vol%'."""
    if pd.isna(val):
        return (np.nan, np.nan)
    txt = str(val)
    wt_match = re.search(r"([\d\.]+)\s*wt", txt, re.I)
    vol_match = re.search(r"([\d\.]+)\s*vol", txt, re.I)
    wt = float(wt_match.group(1)) if wt_match else np.nan
    vol = float(vol_match.group(1)) if vol_match else np.nan
    return wt, vol


def parse_strength(val):
    """Take first numeric token from a strength cell."""
    if pd.isna(val):
        return np.nan
    m = re.match(r"([\d\.]+)", str(val).strip())
    return float(m.group(1)) if m else np.nan


def extract_morpho_flags(txt):
    """
    Very simple keyword-based morphology descriptors
    (size/distribution/shape proxies from the filler text).
    """
    t = str(txt).lower()
    return {
        "has_nanofiller": int(("nano" in t) or ("nanofiller" in t) or ("nanocluster" in t)),
        "has_cluster": int("cluster" in t),
        "has_quartz": int("quartz" in t),
        "has_glass": int("glass" in t),
        "has_zirconia": int(("zirconia" in t) or ("zro" in t)),
        "has_silica": int("silica" in t),
        "has_fiber": int(("fiber" in t) or ("fibrous" in t)),
    }


# ----------------------------
# Load & Feature Engineering
# ----------------------------
print("Loading and preprocessing data...")
# raw = pd.read_csv(INPUT_CSV)
raw = pd.read_excel(INPUT_XLSX)
df = raw.copy()

# Filler wt% / vol%
if "Filler percentage" in df.columns:
    df[["filler_wt", "filler_vol"]] = df["Filler percentage"].apply(
        lambda v: pd.Series(parse_filler_percent(v))
    )
else:
    df["filler_wt"] = np.nan
    df["filler_vol"] = np.nan

# Clean Type and one-hot encode
if "Type" in df.columns:
    df["Type_clean"] = df["Type"].astype(str).str.strip().str.lower()
    type_dummies = pd.get_dummies(df["Type_clean"], prefix="type")
else:
    type_dummies = pd.DataFrame(index=df.index)

# Morphology flags from Filler description
if "Filler" in df.columns:
    morpho_flags = df["Filler"].apply(extract_morpho_flags).apply(pd.Series)
else:
    morpho_flags = pd.DataFrame(index=df.index)

# Target
if "Flexure strength Mpa" in df.columns:
    df["flexural_strength_mpa"] = df["Flexure strength Mpa"].apply(parse_strength)
else:
    cand = [c for c in df.columns if "strength" in c.lower()]
    if len(cand) == 0:
        raise ValueError("Could not find a flexural strength column.")
    df["flexural_strength_mpa"] = df[cand[0]].apply(parse_strength)

df.dropna(subset=[target], inplace=True)
df_eng = pd.concat([df, type_dummies, morpho_flags], axis=1)
df_eng["product_id"] = [f"prod_{i:03d}" for i in range(len(df_eng))]

# ----------------------------
# Data Augmentation
# ----------------------------
print("Augmenting data...")
continuous_cols = [c for c in ["filler_wt", "filler_vol"] if c in df_eng.columns]
binary_cols = list(morpho_flags.columns) + list(type_dummies.columns)

aug_rows = []
for idx, row in df_eng.iterrows():
    base = row.copy()
    base["is_augmented"] = 0
    base["source_index"] = idx
    aug_rows.append(base)

    for _ in range(N_AUG):
        aug = row.copy()
        for c in continuous_cols:
            if pd.notna(row[c]):
                aug[c] = float(row[c]) * np.random.uniform(0.95, 1.05)
        for c in binary_cols:
            if c not in row or pd.isna(row[c]):
                continue
            if float(row[c]) <= 0.5:
                aug[c] = np.round(np.random.uniform(0.0, 0.1), 3)
            else:
                aug[c] = np.round(np.random.uniform(0.95, 1.05), 3)
        aug["is_augmented"] = 1
        aug["source_index"] = idx
        aug_rows.append(aug)

aug = pd.DataFrame(aug_rows)
aug.dropna(subset=[target], inplace=True)
aug.to_csv(AUG_CSV, index=False)
print(f"Saved augmented dataset -> {AUG_CSV} | shape={aug.shape}")

# ----------------------------
# Prepare Data for Modeling & Save
# ----------------------------
print("Preparing data for modeling...")
group_col = "source_index"
drop_cols = [
    "Composite", "Manufacturer", "Resin Monomer", "Filler", "Filler percentage",
    "Type", "Type_clean", "Flexure strength Mpa", target, group_col,
    "is_augmented", "product_id",
]
X_df = aug.drop(columns=[c for c in drop_cols if c in aug.columns])
X_df = X_df.select_dtypes(include=[np.number]).copy()
feature_names = X_df.columns.tolist()
y = aug[target].astype(float).values
groups = aug[group_col].values
X = X_df.values
col_medians = np.nanmedian(X, axis=0)
inds = np.where(np.isnan(X))
X[inds] = np.take(col_medians, inds[1])

joblib.dump({
    'X': X,
    'y': y,
    'groups': groups,
    'feature_names': feature_names,
    'target': target,
    'SEED': SEED
}, PROCESSED_DATA_FILE)

print(f"Saved processed data to -> {PROCESSED_DATA_FILE}")