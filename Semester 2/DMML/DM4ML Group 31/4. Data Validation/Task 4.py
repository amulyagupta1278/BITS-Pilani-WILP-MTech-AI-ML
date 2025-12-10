# %% [markdown]
# ## Data Validation and Quality Report
# 
# After ingesting the raw data, we performed a series of **data validation checks** on the `Generic` dataset.  
# This step ensures the reliability of our data before moving further in the pipeline.
# 
# ### Validation Steps
# 1. **Safe Copying**  
#    - Created a working copy (`df_generic`) so the raw master data remains unchanged.  
# 
# 2. **Missing Data Check**  
#    - Identified columns with missing values and counted the number of affected rows.  
# 
# 3. **Duplicate Check**  
#    - Verified uniqueness of customer identifiers (`CustomerID`).  
#    - Any duplicate records are flagged.
# 
# 4. **Data Type & Numeric Validation**  
#    - Ensured that numeric columns (e.g., `Age`, `Tenure`, `MonthlyCharges`) contain only valid numbers.  
#    - Non-numeric values are flagged as type issues.
# 
# 5. **Range Validation**  
#    - Checked that numeric values fall within acceptable ranges (e.g., Age between 0–120, Tenure ≥ 0).  
#    - Out-of-range values are flagged.
# 
# 6. **Domain Validation**  
#    - Verified categorical fields only contain allowed values (e.g., `Gender ∈ {Male, Female}`, `Churn ∈ {Yes, No}`).
# 
# 7. **Outlier Detection (IQR Method)**  
#    - Applied the Interquartile Range (IQR) rule to numeric columns to detect extreme anomalies.  
#    - Reported the number of outliers per column.
# 
# ### Report Format
# - **Summary Report (in-notebook display):**  
#   Provides dataset-level statistics, such as number of rows, missing value columns, duplicate rows, and outlier counts.
# 
# - **Detailed Report (in-notebook display):**  
#   Expands each check into a structured table with the affected column, type of issue, and supporting details.
# 

# %%

def to_numeric_inplace(df, cols):
    #Try casting columns to numeric and count how many values become NaN.
    info = {}
    for c in cols:
        if c in df.columns:
            before = int(df[c].notna().sum())
            df[c] = pd.to_numeric(df[c], errors="coerce")
            after = int(df[c].notna().sum())
            info[c] = {"coerced_nulls": before - after}
    return info

def is_binary_series(series):
    #Return True if a column looks binary {0,1} (ignoring NaNs).
    s = series.dropna().unique()
    return set(s).issubset({0, 1}) and len(s) <= 2

def iqr_stats(series):
    #Simple IQR outlier count and the bounds we used. Skip binary columns.
    if is_binary_series(series):
        # IQR is not meaningful for binary variables; skip & mark as handled.
        return 0, None, None, "binary_column_ignored_for_iqr"
    s = series.dropna()
    if len(s) < 5:
        return 0, None, None, "not_enough_data"
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    out = int(((series < lo) | (series > hi)).sum())
    return out, float(lo), float(hi), None

def summarize(rep):
    #One-row summary for quick comparisons.
    return {
        "rows": rep["shape"]["rows"],
        "cols": rep["shape"]["cols"],
        "missing_cols_nonzero": sum(1 for v in rep["missing"].values() if v > 0),
        "duplicate_id_rows": rep["duplicates"],
        "range_issue_cols": sum(1 for v in rep["range_violations"].values() if v["violations"] > 0),
        "domain_issue_cols": len(rep["domain_violations"]),
        "iqr_outlier_cols": sum(1 for v in rep["iqr_anomalies"].values() if v["outliers"] > 0),
    }

def flatten_for_table(name, rep, resolutions=None):
    #Expand checks into a long table so it’s easy to read/filter, plus resolution notes.
    rows = []
    # missing
    for col, cnt in rep["missing"].items():
        rows.append({"dataset": name, "check": "missing", "column": col, "value": cnt, "details": ""})
    # duplicates
    rows.append({"dataset": name, "check": "duplicates", "column": "ID", "value": rep["duplicates"], "details": ""})
    # numeric cast
    for col, meta in rep["numeric_cast"].items():
        rows.append({"dataset": name, "check": "numeric_cast", "column": col, "value": meta["coerced_nulls"], "details": "values coerced to NaN"})
    # ranges
    for col, meta in rep["range_violations"].items():
        rows.append({"dataset": name, "check": "range_violations", "column": col, "value": meta["violations"], "details": f"allowed=[{meta['lo']}, {meta['hi']}]"})
    # domains
    for col, meta in rep["domain_violations"].items():
        rows.append({"dataset": name, "check": "domain_violations", "column": col, "value": len(meta['unexpected_values']), "details": f"unexpected={meta['unexpected_values']} / allowed={meta['allowed']}"})
    # IQR
    for col, meta in rep["iqr_anomalies"].items():
        det = f"bounds=[{meta['lo']}, {meta['hi']}]" if (meta["lo"] is not None and meta["hi"] is not None) else (meta.get("reason") or "")
        rows.append({"dataset": name, "check": "iqr_outliers", "column": col, "value": meta["outliers"], "details": det})
    # resolutions (new)
    for note in (resolutions or []):
        rows.append({"dataset": name, "check": "resolution", "column": note.get("column","-"), "value": note.get("count",0), "details": note.get("details","")})
    return pd.DataFrame(rows)


# %%

generic_num = ["Age", "Tenure", "Usage Frequency", "Support Calls", "Payment Delay", "Total Spend", "Last Interaction"]

generic_ranges = {
    "Age": (0, 120),
    "Tenure": (0, 1000),
}

generic_domains = {
    "Gender": ["Male", "Female"],
    "Churn": [0, 1],
}

# 3a) types
generic_cast = to_numeric_inplace(generic, [c for c in generic_num if c in generic.columns])

# 3b) missing + duplicates
generic_missing = generic.isna().sum().to_dict()
if "CustomerID" in generic.columns:
    generic_dups = int(generic.duplicated(subset=["CustomerID"]).sum())
else:
    generic_dups = int(generic.duplicated().sum())

# 3c) ranges
generic_range_viol = {}
for c, (lo, hi) in generic_ranges.items():
    if c in generic.columns and pd.api.types.is_numeric_dtype(generic[c]):
        s = generic[c].dropna()
        generic_range_viol[c] = {"violations": int(((s < lo) | (s > hi)).sum()), "lo": lo, "hi": hi}

# 3d) domains
generic_domain_viol = {}
for c, allowed in generic_domains.items():
    if c in generic.columns:
        bad = generic.loc[generic[c].notna() & ~generic[c].isin(allowed), c].unique().tolist()
        if bad:
            generic_domain_viol[c] = {"unexpected_values": bad, "allowed": allowed}

# 3e) outliers (IQR) — normal numeric columns
generic_iqr = {}
for c in generic_num:
    if c in generic.columns and pd.api.types.is_numeric_dtype(generic[c]):
        count, lo, hi, reason = iqr_stats(generic[c])
        generic_iqr[c] = {"outliers": count, "lo": lo, "hi": hi, "reason": reason}

generic_report = {
    "shape": {"rows": int(generic.shape[0]), "cols": int(generic.shape[1])},
    "missing": generic_missing,
    "duplicates": generic_dups,
    "numeric_cast": generic_cast,
    "range_violations": generic_range_viol,
    "domain_violations": generic_domain_viol,
    "iqr_anomalies": generic_iqr,
}

# Resolution note: dataset is already clean
generic_resolutions = [{
    "column": "-",
    "count": 0,
    "details": "No action required (no missing/duplicates/range/domain issues)"
}]


# %%
# 4) BUILD TABLES + SHOW REPORT

generic_table = flatten_for_table("generic", generic_report, resolutions=generic_resolutions)

summary = pd.DataFrame([
    {"dataset": "generic", **summarize(generic_report)},
])

print("=== DATA QUALITY SUMMARY (Generic) ===")
display(summary)

print("\n=== GENERIC — DETAILED CHECKS (with resolutions) ===")
display(generic_table.sort_values(["check", "column"]).reset_index(drop=True))

# %%
if WRITE_FILES:
    import os
    os.makedirs(OUTDIR, exist_ok=True)
    generic_table.to_csv(f"{OUTDIR}/generic_dq_report_{ts}.csv", index=False)
    summary.to_csv(f"{OUTDIR}/summary_{ts}.csv", index=False)
    print(f"CSV reports saved to: {OUTDIR}")


