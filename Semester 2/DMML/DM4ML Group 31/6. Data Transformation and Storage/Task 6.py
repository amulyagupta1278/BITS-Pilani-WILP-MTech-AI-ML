# %% [markdown]
# ## 1. Feature Engineering  
# 
# We create new attributes that capture useful business signals, such as:  
# 
# - **Spend per Month**  
#   - Generic: `TotalSpend ÷ Tenure`  
# 
# - **Support Calls per Month**
#   - `Support Calls ÷ Tenure`  
# 
# These derived features provide richer signals for predictive modeling than raw columns.  
# 
# ---
# 

# %%

def _first_present(df: pd.DataFrame, candidates: list[str]):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _safe_div(num, den):
    den = np.where(den == 0, np.nan, den)
    out = num / den
    return np.where(np.isfinite(out), out, 0)

    # payment_punctuality
    if col_pay_delay:
        m = t[col_pay_delay].max()
        m = m if pd.notna(m) and m != 0 else 1
        t["payment_punctuality"] = 1.0 - (t[col_pay_delay].astype(float) / m)
        created.append("payment_punctuality")

    # services_count
    if svc_cols:
        numeric_flags = t[svc_cols].select_dtypes(include=[np.number])
        if numeric_flags.shape[1] > 0:
            t["services_count"] = numeric_flags.sum(axis=1)
            created.append("services_count")

    return t, created
import pandas as pd
import numpy as np

def build_generic_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Feature builder for the Generic dataset only.

    Creates:
      - months_active            (from Tenure)
      - spend_per_month          (MonthlyCharges OR TotalSpend/Tenure)
      - usage_per_month          (UsageFrequency/Tenure)
      - support_calls_per_month  (SupportCalls/Tenure)
      - payment_punctuality      (1 - normalized PaymentDelay)
      - services_count           (sum of service/payment one-hots & flags)
      - tenure_bucket            (new/mid/loyal)
      - last_interaction_year, last_interaction_month, days_since_last_interaction
    """
    def _fp(t, names):
        for n in names:
            if n in t.columns:
                return n
        return None

    def _safe_div(a, b):
        b = np.where(pd.isna(b) | (b == 0), 1, b)
        return a / b

    t = df.copy()
    created = []

    # Aliases for the Generic dataset
    col_tenure   = _fp(t, ["Tenure"])
    col_monthly  = _fp(t, ["MonthlyCharges", "AvgMonthlySpend", "AverageMonthlySpend"])
    col_tspend   = _fp(t, ["Total Spend", "TotalSpend"])
    col_usage    = _fp(t, ["Usage Frequency", "UsageFrequency", "Usage"])
    col_support  = _fp(t, ["Support Calls", "SupportCalls"])
    col_pdelay   = _fp(t, ["Payment Delay", "PaymentDelay", "AvgPaymentDelay"])
    col_lastint  = _fp(t, ["Last Interaction", "LastInteraction", "last_interaction"])

    # Service / product / payment flags (numeric after encoding or pre-0/1)
    service_like_patterns = [
        "Subscription Type", "Subscription_Type", "Product", "Plan",
        "AddOn", "Add-On", "InternetService", "PaymentMethod", "AutoPay",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "MultipleLines", "PhoneService"
    ]
    svc_cols = [c for c in t.columns if any(p in c for p in service_like_patterns)]

    # months_active + tenure_bucket
    if col_tenure:
        t["months_active"] = pd.to_numeric(t[col_tenure], errors="coerce")
        created.append("months_active")
        t["tenure_bucket"] = pd.cut(
            t["months_active"], bins=[-1, 6, 24, np.inf],
            labels=["new(<=6m)", "mid(<=24m)", "loyal(>24m)"]
        )
        created.append("tenure_bucket")

    # spend_per_month
    if col_monthly:
        t["spend_per_month"] = pd.to_numeric(t[col_monthly], errors="coerce")
        created.append("spend_per_month")
    elif col_tspend and col_tenure:
        num = pd.to_numeric(t[col_tspend], errors="coerce")
        den = np.maximum(pd.to_numeric(t[col_tenure], errors="coerce"), 1)
        t["spend_per_month"] = _safe_div(num, den)
        created.append("spend_per_month")

    # usage_per_month
    if col_usage and col_tenure:
        num = pd.to_numeric(t[col_usage], errors="coerce")
        den = np.maximum(pd.to_numeric(t[col_tenure], errors="coerce"), 1)
        t["usage_per_month"] = _safe_div(num, den)
        created.append("usage_per_month")

    # support_calls_per_month
    if col_support and col_tenure:
        num = pd.to_numeric(t[col_support], errors="coerce")
        den = np.maximum(pd.to_numeric(t[col_tenure], errors="coerce"), 1)
        t["support_calls_per_month"] = _safe_div(num, den)
        created.append("support_calls_per_month")

    # payment_punctuality (1 - normalized PaymentDelay)
    if col_pdelay:
        pdv = pd.to_numeric(t[col_pdelay], errors="coerce")
        m = np.nanmax(pdv.values) if np.isfinite(np.nanmax(pdv.values)) and np.nanmax(pdv.values) != 0 else 1.0
        t["payment_punctuality"] = 1.0 - (pdv / m)
        created.append("payment_punctuality")

    # services_count (sum numeric service-like columns)
    if svc_cols:
        num_flags = t[svc_cols].select_dtypes(include=[np.number]).fillna(0.0)
        if num_flags.shape[1] > 0:
            t["services_count"] = num_flags.sum(axis=1)
            created.append("services_count")

    # Date-time features from Last Interaction
    if col_lastint:
        ts = pd.to_datetime(t[col_lastint], errors="coerce")
        t["last_interaction_year"] = ts.dt.year
        t["last_interaction_month"] = ts.dt.month
        ref = ts.max() if ts.notna().any() else pd.Timestamp.now()
        t["days_since_last_interaction"] = (ref - ts).dt.days
        created += ["last_interaction_year", "last_interaction_month", "days_since_last_interaction"]

    return t, created


# %% [markdown]
# ## 2. Scaling and Normalization  
# 
# - All numerical features are **standardized** to bring them into comparable ranges.  
# - This avoids high-magnitude variables (e.g., `Total Spend`) from dominating during model training.  
# 
# ---
# 
# ## 3. Aggregation and Bucketing  
# 
# - Continuous variables like **Tenure** are grouped into **buckets** (e.g., short-term vs. mid-term vs. long-term customers).  
# - Buckets simplify interpretation and help uncover patterns in churn behavior.  
# 
# ---

# %%
from sklearn.preprocessing import MinMaxScaler
generic_feat, gen_created = build_generic_features(generic_df)

def add_scaled(df: pd.DataFrame, candidates: list[str], suffix="_z"):
    cols = [c for c in candidates if c in df.columns]
    if not cols:
        return df, []
    scaler = StandardScaler()
    z = scaler.fit_transform(df[cols])
    df = df.copy()
    out_cols = []
    for i, c in enumerate(cols):
        zc = f"{c}{suffix}"
        df[zc] = z[:, i]
        out_cols.append(zc)
    return df, out_cols


def add_minmax(df: pd.DataFrame, candidates: list[str], suffix="_minmax"):
    cols = [c for c in candidates if c in df.columns]
    if not cols:
        return df, []
    mm = MinMaxScaler()
    m = mm.fit_transform(df[cols])
    df = df.copy()
    out_cols = []
    for i, c in enumerate(cols):
        mc = f"{c}{suffix}"
        df[mc] = m[:, i]
        out_cols.append(mc)
    return df, out_cols

mm_candidates = [c for c in ["months_active","spend_per_month","usage_per_month",
                             "support_calls_per_month","payment_punctuality","services_count"]
                 if c in generic_feat.columns]

generic_feat, gen_minmax = add_minmax(generic_feat, mm_candidates)
print("MinMax scaled:", gen_minmax)

# Need to define which columns are continuous here if build_common_features is not used to identify them
generic_continuous = ["Age", "Tenure", "Usage Frequency", "Support Calls", "Payment Delay", "Total Spend", "spend_per_month", "usage_per_month", "support_calls_per_month", "payment_punctuality", "services_count", "days_since_last_interaction"] # Example: specify columns to scale

generic_feat, gen_scaled = add_scaled(generic_feat.copy(), generic_continuous) # Use generic_feat and capture output

# add a simple row_id key for relational storage
generic_feat.insert(0, "row_id", np.arange(1, len(generic_feat)+1))

def aggregate_generic(df: pd.DataFrame):
    if "CustomerID" in df.columns:
        grp = (df.groupby("CustomerID", as_index=False)
                 .agg(spend_sum=("spend_per_month","sum"),
                      calls_mean=("support_calls_per_month","mean")))
        return grp
    return pd.DataFrame()

generic_agg = aggregate_generic(generic_feat)
if not generic_agg.empty:
    generic_feat = generic_feat.merge(generic_agg, on="CustomerID", how="left")


print("Created features:")
print("  Generic :", gen_created,   "| Scaled:", gen_scaled)

# %%
GEN_FEATS_PATH   = os.path.join(TRANS_GENERIC_DIR, f"generic_features_{TS}.csv")

generic_feat.to_csv(GEN_FEATS_PATH, index=False)

print("Saved transformed feature table:")
print("  generic→", GEN_FEATS_PATH)

# %%
# Pivot example
pivot_tbl = pd.DataFrame()
if {"CustomerID","Subscription Type","Usage"}.issubset(set(generic.columns)):
    pivot_tbl = (generic.pivot_table(index="CustomerID",
                                     columns="Subscription Type",
                                     values="Usage",
                                     aggfunc="sum",
                                     fill_value=0)
                          .reset_index())
    generic_feat = generic_feat.merge(pivot_tbl, on="CustomerID", how="left")

# Unpivot example
wide_candidates = [c for c in ["months_active","spend_per_month","usage_per_month"] if c in generic_feat.columns]
if wide_candidates:
    long_tbl = generic_feat.melt(id_vars=["row_id"],
                                 value_vars=wide_candidates,
                                 var_name="metric",
                                 value_name="value")

# %% [markdown]
# ## 4. Data Storage  
# 
# - Transformed datasets are written into a **relational database (SQLite)**.  
# - Benefits:  
#   - Efficient querying (e.g., churn rates by tenure bucket).  
#   - Reusability across modeling steps.  
#   - Serves as a foundation for a future **Feature Store**, where features are versioned and shared across training and inference pipelines.  
# 

# %%
engine = create_engine(DB_URL, future=True)

# Write full dataframes (all columns) to DB
generic_feat.to_sql("generic_features", engine, if_exists="replace", index=False)

# Pull the CREATE TABLE DDL from sqlite_master and save to a .sql file (deliverable)
with engine.connect() as conn:
    ddl_generic = conn.execute(text("SELECT sql FROM sqlite_master WHERE type='table' AND name='generic_features';")).scalar()

SCHEMA_PATH = os.path.join(DB_DIR, f"schema_{TS}.sql")
with open(SCHEMA_PATH, "w") as f:
    f.write("-- Auto-extracted schema for SQLite tables\n\n")
    f.write((ddl_generic or "-- generic_features not found") + ";\n")

# Sample queries (deliverable)
SAMPLE_QUERIES = textwrap.dedent(f"""
-- SAMPLE QUERIES (SQLite)

-- 1) Generic: average spend_per_month and churn
SELECT ROUND(AVG(spend_per_month), 2) AS avg_spm,
       ROUND(AVG(CAST({TARGET_COL} AS FLOAT)), 4) AS churn_rate,
       COUNT(*) AS n
FROM generic_features;

-- 2) Generic: support burden vs churn (bucketed)
SELECT CASE
         WHEN support_calls_per_month < 0.25 THEN 'low'
         WHEN support_calls_per_month < 0.75 THEN 'med'
         ELSE 'high'
       END AS support_load,
       COUNT(*) AS n,
       ROUND(AVG(CAST({TARGET_COL} AS FLOAT)), 4) AS churn_rate
FROM generic_features
GROUP BY support_load
ORDER BY n DESC;
""").strip()

QUERIES_PATH = os.path.join(DB_DIR, f"sample_queries_{TS}.sql")
with open(QUERIES_PATH, "w") as f:
    f.write(SAMPLE_QUERIES + "\n")

print("Database written to:", DB_PATH)
print("Schema saved to    :", SCHEMA_PATH)
print("Queries saved to   :", QUERIES_PATH)

# %%
with engine.connect() as conn:
    q4 = conn.execute(text(f"""
        SELECT ROUND(AVG(spend_per_month), 2) AS avg_spm,
               ROUND(AVG(CAST({TARGET_COL} AS FLOAT)), 4) AS churn_rate,
               COUNT(*) AS n
        FROM generic_features;
    """)).fetchall()
    print("\nGeneric — spend_per_month + churn:")
    for r in q4: print(r._asdict())

    q5 = conn.execute(text(f"""
        SELECT CASE
                 WHEN support_calls_per_month < 0.25 THEN 'low'
                 WHEN support_calls_per_month < 0.75 THEN 'med'
                 ELSE 'high'
               END AS support_load,
               COUNT(*) AS n,
               ROUND(AVG(CAST({TARGET_COL} AS FLOAT)), 4) AS churn_rate
        FROM generic_features
        GROUP BY support_load
        ORDER BY n DESC;
    """)).fetchall()
    print("\nGeneric — support burden vs churn (bucketed):")
    for r in q5: print(r._asdict())

# %%
with engine.connect() as conn:
    q4 = conn.execute(text(f"""
        SELECT ROUND(AVG(spend_per_month), 2) AS avg_spm,
               ROUND(AVG(CAST({TARGET_COL} AS FLOAT)), 4) AS churn_rate,
               COUNT(*) AS n
        FROM generic_features;
    """)).fetchall()
    print("\nGeneric — spend_per_month + churn:")
    for r in q4: print(r._asdict())

    q5 = conn.execute(text(f"""
        SELECT CASE
                 WHEN support_calls_per_month < 0.25 THEN 'low'
                 WHEN support_calls_per_month < 0.75 THEN 'med'
                 ELSE 'high'
               END AS support_load,
               COUNT(*) AS n,
               ROUND(AVG(CAST({TARGET_COL} AS FLOAT)), 4) AS churn_rate
        FROM generic_features
        GROUP BY support_load
        ORDER BY n DESC;
    """)).fetchall()
    print("\nGeneric — support burden vs churn (bucketed):")
    for r in q5: print(r._asdict())

# %% [markdown]
# ## Transformation Summary
# 
# The transformation process created consistent and comparable features for dataset.
# The engineered features mainly focused on **customer activity (tenure, usage, support calls)**,
# **spending behavior (monthly spend, payment punctuality)**, and **service adoption (count of subscribed services)**.
# 
# Key observations:
# - **Generic dataset**:
#   - Similar transformations applied (`months_active`, `tenure_bucket`, `spend_per_month`).
#   - Additional engineered metrics include `usage_per_month`, `support_calls_per_month`, and `payment_punctuality`.
#   - Standardized versions of each were created to normalize scales.
# 
# The resulting summaries have been stored in the warehouse:
# 
# - **Generic:** `/content/drive/MyDrive/DMML_Project/data/warehouse/sqlite/generic_transformation_summary_20250819T143634.csv`
# 
# These transformations ensure the dataset is aligned for downstream modeling while preserving its unique characteristics.


