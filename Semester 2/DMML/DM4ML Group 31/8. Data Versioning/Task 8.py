# %% [markdown]
# # Data Versioning
# This step implements **data versioning** for feature snapshots.  
# - Tracks schema changes (added/removed features)  
# - Compares sample rows between `v1` and `v2`  
# - Maintains metadata logs for reproducibility and audit trail  
# - Generates a short report + zipped submission proof  

# %%

import os
import re
import pandas as pd
import numpy as np

# --- Paths & naming ---
FS_BASE = "/content/drive/MyDrive/DMML_Project/data/feature_store"
DATA_DIR = f"{FS_BASE}/feature_data"
META_PATH = f"{FS_BASE}/feature_metadata.csv"
SNAPSHOT_PREFIX = "generic_features"   # from Point 7
SNAP_DATE = "2025-06-30"               # optional fixed date; we also auto-detect latest if not present
TARGET_COL = "Churn" # Define TARGET_COL here

os.makedirs(DATA_DIR, exist_ok=True)

# --- Helper: find the v1 snapshot (use given SNAP_DATE if file exists; else pick latest *_v1.csv) ---
def find_v1_snapshot(prefix: str, snap_date: str) -> str:
    candidate = os.path.join(DATA_DIR, f"{prefix}_{snap_date}_v1.csv")
    if os.path.exists(candidate):
        return candidate

    # fallback: pick latest *_v1.csv by lexical sort
    files = [f for f in os.listdir(DATA_DIR) if f.startswith(prefix) and f.endswith("_v1.csv")]
    if not files:
        raise FileNotFoundError(f"No v1 snapshots found in {DATA_DIR} for prefix '{prefix}'.")
    files.sort()
    return os.path.join(DATA_DIR, files[-1])

v1_path = find_v1_snapshot(SNAPSHOT_PREFIX, SNAP_DATE)
print("Using v1 snapshot:", v1_path)

# --- Load v1 (no event_timestamp in generic flow) ---
v1 = pd.read_csv(v1_path)

# --- Create v2 with realistic changes for the generic schema ---
# 1) New feature: services_per_tenure
#    Defensive divide-by-zero: use max(tenure, 1)
v2 = v1.copy()
if "services_count" not in v2.columns or "tenure" not in v2.columns:
    raise KeyError("Expected columns 'services_count' and 'tenure' were not found in the snapshot.")

den = v2["tenure"].fillna(0).clip(lower=1).astype(float)
v2["services_per_tenure"] = (v2["services_count"].astype(float) / den).round(4)

# 2) Tweak existing feature logic (simulate new v2 transformation):
#    If PhoneService == 1, bump services_count_minmax by +0.05 (then clip to [0, 1]).
if "services_count_minmax" in v2.columns and "PhoneService" in v2.columns:
    # Ensure PhoneService is numeric before comparison
    v2['PhoneService'] = pd.to_numeric(v2['PhoneService'], errors='coerce').fillna(0).astype(int)
    bump = (v2["PhoneService"].astype(float) == 1.0).astype(float) * 0.05
    v2["services_count_minmax"] = (v2["services_count_minmax"].astype(float) + bump).clip(0.0, 1.0)

# --- Compose v2 path using the same date as v1 (parsed from v1 filename) ---
m = re.search(rf"{SNAPSHOT_PREFIX}_(\d{{4}}-\d{{2}}-\d{{2}})_v1\.csv$", os.path.basename(v1_path))
snap_date_from_v1 = m.group(1) if m else SNAP_DATE
v2_path = os.path.join(DATA_DIR, f"{SNAPSHOT_PREFIX}_{snap_date_from_v1}_v2.csv")

# --- Save v2 snapshot ---
v2.to_csv(v2_path, index=False)
print("Created v2 snapshot:", v2_path)

# --- Bump metadata versions & add new feature row ---
if not os.path.exists(META_PATH):
    raise FileNotFoundError(f"Feature metadata not found at {META_PATH}")

meta = pd.read_csv(META_PATH)

def upsert(meta_df: pd.DataFrame, name: str, dtype: str, desc: str, src: str, ver: str) -> pd.DataFrame:
    if (meta_df["feature_name"] == name).any():
        meta_df.loc[meta_df["feature_name"] == name, ["dtype","description","source","version"]] = [dtype, desc, src, ver]
        return meta_df
    return pd.concat([meta_df, pd.DataFrame([{
        "feature_name": name,
        "dtype": dtype,
        "description": desc,
        "source": src,
        "version": ver
    }])], ignore_index=True)

# Bump existing feature changed by v2 logic
meta = upsert(
    meta,
    name="services_count_minmax",
    dtype="float",
    desc="Min-max scaled services_count (v2 logic: +0.05 if PhoneService==1, clipped to [0,1])",
    src="transformation_v2",
    ver="2.0.0"
)

# Add the new derived feature
meta = upsert(
    meta,
    name="services_per_tenure",
    dtype="float",
    desc="Derived: services_count / max(tenure, 1) (v2)",
    src="transformation_v2",
    ver="1.0.0"
)

# Persist updated metadata
meta.sort_values("feature_name").to_csv(META_PATH, index=False)

print("Updated metadata with version bumps.")
display(meta.head(10))

# %%

import os
import re
import pandas as pd

# --- Paths & naming ---
FS_BASE = "/content/drive/MyDrive/DMML_Project/data/feature_store"
DATA_DIR = f"{FS_BASE}/feature_data"
META_PATH = f"{FS_BASE}/feature_metadata.csv"
SNAPSHOT_PREFIX = "generic_features"
SNAP_DATE = "2025-06-30"   # if not found, we'll auto-pick the latest by name

os.makedirs(f"{FS_BASE}/reports", exist_ok=True)

# --- Helper: build snapshot path by date & suffix ---
def snapshot_path(prefix: str, snap_date: str, v: str) -> str:
    return os.path.join(DATA_DIR, f"{prefix}_{snap_date}_{v}.csv")

# --- If explicit date files don't exist, detect latest matching v1/v2 ---
def detect_latest(prefix: str, version_suffix: str) -> str:
    files = [f for f in os.listdir(DATA_DIR) if f.startswith(prefix) and f.endswith(f"_{version_suffix}.csv")]
    if not files:
        raise FileNotFoundError(f"No snapshots found for suffix {version_suffix} in {DATA_DIR}")
    files.sort()
    return os.path.join(DATA_DIR, files[-1])

v1_path = snapshot_path(SNAPSHOT_PREFIX, SNAP_DATE, "v1")
v2_path = snapshot_path(SNAPSHOT_PREFIX, SNAP_DATE, "v2")

if not os.path.exists(v1_path):
    v1_path = detect_latest(SNAPSHOT_PREFIX, "v1")
if not os.path.exists(v2_path):
    v2_path = detect_latest(SNAPSHOT_PREFIX, "v2")

print("Using v1:", v1_path)
print("Using v2:", v2_path)

# --- Load snapshots (no event_timestamp in generic flow) ---
v1 = pd.read_csv(v1_path)
v2 = pd.read_csv(v2_path)
meta = pd.read_csv(META_PATH)

# --- Column diffs ---
cols_v1 = set(v1.columns)
cols_v2 = set(v2.columns)
added_cols = sorted(list(cols_v2 - cols_v1))
removed_cols = sorted(list(cols_v1 - cols_v2))
common_cols = sorted(list(cols_v1 & cols_v2))

# --- Sample diff on the feature changed in v2 logic ---
# We tweaked services_count_minmax in Cell 4 (+0.05 if PhoneService==1, clip to [0,1])
sample_key = "row_id"
compare_col = "services_count_minmax" if "services_count_minmax" in common_cols else common_cols[0]

sample_ids = v1[sample_key].astype(str).head(5).tolist()
merged = (
    v1[v1[sample_key].astype(str).isin(sample_ids)][[sample_key, compare_col]]
    .merge(
        v2[v2[sample_key].astype(str).isin(sample_ids)][[sample_key, compare_col]],
        on=sample_key,
        suffixes=("_v1","_v2")
    )
)

# --- Build report text ---
lines = []
lines.append("=== Point 8: Feature Versioning Demo ===")
lines.append(f"v1 rows={len(v1)}, v2 rows={len(v2)}")
lines.append(f"Added columns in v2: {added_cols}")
lines.append(f"Removed columns in v2: {removed_cols}")
lines.append(f"\nSample {compare_col} change (first 5 rows):")
lines.append(merged.to_string(index=False))
lines.append("\nFeature metadata (head):")
lines.append(meta.head(10).to_string(index=False))

# --- Save proof report ---
out_txt = f"{FS_BASE}/reports/versioning_demo_output.txt"
with open(out_txt, "w") as f:
    f.write("\n".join(lines))

print(f"Wrote {out_txt}")
print("\n".join(lines[:12]))



