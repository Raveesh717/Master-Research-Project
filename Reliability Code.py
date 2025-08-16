# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 10:30:58 2025

@author: RY163UL
"""
import pandas as pd
import numpy as np
from pathlib import Path

# ---------- Reliability helpers ----------
def cronbach_alpha(df: pd.DataFrame) -> float:
    df = df.dropna()
    k = df.shape[1]
    if k < 2:
        return np.nan
    # if any column is constant, variance is 0 -> alpha can still compute;
    # pandas var handles this; total_var 0 => NaN
    item_vars = df.var(axis=0, ddof=1)
    total_var = df.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    return (k / (k - 1.0)) * (1 - item_vars.sum() / total_var)

def avg_inter_item_corr(df: pd.DataFrame) -> float:
    df = df.dropna()
    if df.shape[1] < 2:
        return np.nan
    corr = df.corr()
    iu = np.triu_indices_from(corr, k=1)
    vals = corr.values[iu]
    # Avoid warnings if all-NaN (e.g., constant columns)
    return float(np.nanmean(vals)) if np.isfinite(np.nanmean(vals)) else np.nan

def item_total_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    total = df.sum(axis=1)
    rows = []
    for col in df.columns:
        other_total = total - df[col]
        # handle constant vectors to avoid nan corr warnings
        x = df[col].values
        y = other_total.values
        if np.nanstd(x, ddof=1) == 0 or np.nanstd(y, ddof=1) == 0:
            citc = np.nan
        else:
            citc = np.corrcoef(x, y)[0, 1]
        sub_alpha = cronbach_alpha(df.drop(columns=[col])) if df.shape[1] > 2 else np.nan
        rows.append({"item": col,
                     "corrected_item_total_corr": citc,
                     "alpha_if_deleted": sub_alpha})
    return pd.DataFrame(rows)

def bootstrap_alpha(df: pd.DataFrame, n_boot: int = 2000, seed: int = 123):
    rng = np.random.default_rng(seed)
    df = df.dropna()
    n = len(df)
    if n < 5 or df.shape[1] < 2:
        return (np.nan, np.nan)
    alphas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sample = df.iloc[idx]
        alphas.append(cronbach_alpha(sample))
    return (float(np.nanpercentile(alphas, 2.5)), float(np.nanpercentile(alphas, 97.5)))

# ---------- Path handling ----------
# Leave BASE as None for Windows absolute paths; set to a folder only if using relative files.
BASE: Path | None = None  # e.g., Path(r"C:\Users\RY163UL\Downloads") if you prefer

def resolve(p: str | Path) -> Path:
    p = Path(p)
    # If absolute (like C:\...), return as is. If relative and BASE is set, join BASE/relative.
    if p.is_absolute():
        return p
    if BASE is not None:
        return BASE / p
    return p

paths = {
    "Public": r"C:\Users\RY163UL\Downloads\Public Analysis.xlsx",
    "ENV":    r"C:\Users\RY163UL\Downloads\Env Analysis.xlsx",
    "Trust":  r"C:\Users\RY163UL\Downloads\Trust Analysis.xlsx",
    "TSE":    r"C:\Users\RY163UL\Downloads\TSE Analysis.xlsx",
    "SOC":    r"C:\Users\RY163UL\Downloads\SOC Analysis.xlsx",
    "TAM":    r"C:\Users\RY163UL\Downloads\TAM Analysis.xlsx",
    # "Demo" is optional; if you need it later, add and read similarly.
}

def read_xlsx(path_key: str) -> pd.DataFrame:
    file = resolve(paths[path_key])
    try:
        return pd.read_excel(file)  # engine='openpyxl' if needed
    except FileNotFoundError:
        raise FileNotFoundError(f"{path_key} file not found at: {file}")
    except Exception as e:
        raise RuntimeError(f"Failed to read {path_key} at {file}: {e}")

# Read dataframes
pub_df   = read_xlsx("Public")
env_df   = read_xlsx("ENV")
trust_df = read_xlsx("Trust")
tse_df   = read_xlsx("TSE")
soc_df   = read_xlsx("SOC")
tam_df   = read_xlsx("TAM")

# ---------- Define scales ----------
# Uses all columns except the participant identifier (assumed exact name "Participant No")
def cols_except_id(df: pd.DataFrame, id_col="Participant No"):
    return [c for c in df.columns if c != id_col]

scale_map = {
    "Public": cols_except_id(pub_df),
    "ENV":    cols_except_id(env_df),
    "Trust":  cols_except_id(trust_df),
    "TSE":    cols_except_id(tse_df),
    "SOC":    cols_except_id(soc_df),
    "PU":     [c for c in tam_df.columns if str(c).startswith("PU")],
    "PEOU":   [c for c in tam_df.columns if str(c).startswith("PEOU")],
    "BI/IU":  [c for c in tam_df.columns if str(c).startswith(("IU", "BI"))],
}

def reliability_for(df: pd.DataFrame, cols):
    if not cols:
        return 0, np.nan, (np.nan, np.nan), np.nan, pd.DataFrame(columns=["item","corrected_item_total_corr","alpha_if_deleted"])
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    X = X.dropna()  # listwise deletion for reliability stats
    n = X.shape[0]
    if X.shape[1] < 2 or n < 2:
        return n, np.nan, (np.nan, np.nan), np.nan, item_total_stats(X) if X.shape[1] else pd.DataFrame(columns=["item","corrected_item_total_corr","alpha_if_deleted"])
    a = cronbach_alpha(X)
    aic = avg_inter_item_corr(X)
    low, high = bootstrap_alpha(X, n_boot=2000, seed=123)
    diag = item_total_stats(X)
    return n, a, (low, high), aic, diag

# ---------- Compute ----------
summary_rows = []
diags = {}

for scale, cols in scale_map.items():
    source = (
        pub_df   if scale == "Public" else
        env_df   if scale == "ENV" else
        trust_df if scale == "Trust" else
        tse_df   if scale == "TSE" else
        soc_df   if scale == "SOC" else
        tam_df
    )
    n, alpha, ci, aic, diag = reliability_for(source, cols)
    summary_rows.append({
        "Scale": scale,
        "Items (k)": len(cols),
        "N (listwise)": n,
        "Cronbach's alpha": round(alpha, 3) if pd.notnull(alpha) else np.nan,
        "95% CI (alpha)": f"[{ci[0]:.3f}, {ci[1]:.3f}]" if pd.notnull(ci[0]) else np.nan,
        "Avg inter-item corr": round(aic, 3) if pd.notnull(aic) else np.nan
    })
    diags[scale] = diag

summary = pd.DataFrame(summary_rows).sort_values("Scale")
print(summary.to_string(index=False))

# ---------- Save Excel report ----------
out_path = Path.cwd() / "reliability_report.xlsx"
with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
    summary.to_excel(writer, index=False, sheet_name="Summary")
    for scale, diag in diags.items():
        safe_name = scale.replace("/", "_") + "_items"
        diag_ = diag.copy()
        if not diag_.empty:
            diag_["corrected_item_total_corr"] = diag_["corrected_item_total_corr"].round(3)
            diag_["alpha_if_deleted"] = diag_["alpha_if_deleted"].round(3)
        diag_.to_excel(writer, index=False, sheet_name=safe_name)

print(f"\nSaved: {out_path}")
