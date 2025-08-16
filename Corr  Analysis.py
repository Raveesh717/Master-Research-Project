# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 13:41:23 2025

@author: RY163UL
"""

import pandas as pd, numpy as np
from scipy.stats import pearsonr

# --- Paths (Windows local) ---
paths = {
    "Public": r"C:\Users\RY163UL\Downloads\Public Analysis.xlsx",
    "ENV":    r"C:\Users\RY163UL\Downloads\Env Analysis.xlsx",
    "Trust":  r"C:\Users\RY163UL\Downloads\Trust Analysis.xlsx",
    "TSE":    r"C:\Users\RY163UL\Downloads\TSE Analysis.xlsx",
    "SOC":    r"C:\Users\RY163UL\Downloads\SOC Analysis.xlsx",
    "TAM":    r"C:\Users\RY163UL\Downloads\TAM Analysis.xlsx",
    "Demo":   r"C:\Users\RY163UL\Downloads\Demo Analysis.xlsx"
}

# --- Load and normalize headers ---
pub  = pd.read_excel(paths["Public"]);  pub.columns  = [c.strip() for c in pub.columns]
env_ = pd.read_excel(paths["ENV"]);     env_.columns = [c.strip() for c in env_.columns]
tru  = pd.read_excel(paths["Trust"]);   tru.columns  = [c.strip() for c in tru.columns]
tse  = pd.read_excel(paths["TSE"]);     tse.columns  = [c.strip() for c in tse.columns]
soc  = pd.read_excel(paths["SOC"]);     soc.columns  = [c.strip() for c in soc.columns]
tam  = pd.read_excel(paths["TAM"]);     tam.columns  = [c.strip() for c in tam.columns]
demo = pd.read_excel(paths["Demo"]);    demo.columns = [c.strip() for c in demo.columns]

# --- Item maps (already reversed items) ---
items = {
    "SOC":   (soc,  [c for c in soc.columns if c != "Participant No"]),
    "PU":    (tam,  [c for c in tam.columns if c.startswith("PU")]),
    "PEOU":  (tam,  [c for c in tam.columns if c.startswith("PEOU")]),
    "Adoption Intention": (tam, [c for c in tam.columns if c.startswith("IU")]),
    "TSE":   (tse,  [c for c in tse.columns if c != "Participant No"]),
    "Trust": (tru,  [c for c in tru.columns if c != "Participant No"]),
    "Environment": (env_, [c for c in env_.columns if c != "Participant No"]),
    "Public": (pub, [c for c in pub.columns if c != "Participant No"])
}

def composite_mean(df, cols, min_frac=0.5):
    sub = df[cols].apply(pd.to_numeric, errors="coerce")
    n_ok = sub.notna().sum(axis=1)
    thr = int(np.ceil(len(cols)*min_frac))
    m = sub.mean(axis=1)
    m[n_ok < thr] = np.nan
    return m

# Master index (TAM has full ID list)
master = tam[["Participant No"]].rename(columns={"Participant No":"participant"}).copy()
comp = master.copy()
for name, (df, cols) in items.items():
    s = composite_mean(df, cols, min_frac=0.5)
    comp = comp.merge(pd.DataFrame({"participant": df["Participant No"], name: s}), on="participant", how="left")

# Demographics
demo_m = demo.rename(columns={
    "Participant No": "participant",
    "Please state your age": "age",
    "Which residential location are you from?": "res_location",
    "How would you rate your ability to use digital technology (smartphones, apps, computers)?": "digital_lit",
    "Do you use any “smart” devices or apps to monitor/manage home energy?": "smart_use"
})

def norm_lower(x): 
    if pd.isna(x): return x
    return str(x).strip().lower()

demo_m["age"] = pd.to_numeric(demo_m["age"], errors="coerce")
demo_m["res_location"] = demo_m["res_location"].apply(norm_lower).map({"urban":1, "rural":0})
demo_m["digital_lit"] = demo_m["digital_lit"].apply(norm_lower).map({"low":1, "moderate":2, "high":3})
demo_m["smart_use"] = demo_m["smart_use"].apply(norm_lower).map({"yes":1, "no":0})

comp = comp.merge(demo_m[["participant","age","res_location","digital_lit","smart_use"]], on="participant", how="left")

# Remove Participant 86; confirm N=235
comp = comp[comp["participant"] != 86]

# Variables
vars_core  = ["PU","PEOU","Adoption Intention","SOC","TSE","Trust","Environment","Public"]
vars_extra = ["age","res_location","digital_lit","smart_use"]
vars_all   = vars_core + vars_extra

# Correlation function
def corr_with_p(df, cols):
    r = pd.DataFrame(index=cols, columns=cols, dtype=float)
    p = pd.DataFrame(index=cols, columns=cols, dtype=float)
    n = pd.DataFrame(index=cols, columns=cols, dtype="Int64")
    for i in cols:
        for j in cols:
            mask = df[i].notna() & df[j].notna()
            n_ij = int(mask.sum()); n.loc[i,j] = n_ij
            if n_ij >= 3:
                r_ij, p_ij = pearsonr(df.loc[mask, i], df.loc[mask, j])
                r.loc[i,j], p.loc[i,j] = r_ij, p_ij
            else:
                r.loc[i,j] = np.nan; p.loc[i,j] = np.nan
    return r, p, n

r, p, n = corr_with_p(comp, vars_all)

# Save outputs
with pd.ExcelWriter(r"C:\Users\RY163UL\Downloads\correlations_report.xlsx", engine="xlsxwriter") as writer:
    r.round(3).to_excel(writer, sheet_name="Pearson_r")
    p.applymap(lambda v: np.round(v, 4) if pd.notnull(v) else v).to_excel(writer, sheet_name="p_values")
    n.to_excel(writer, sheet_name="pairwise_N")
    comp.to_excel(writer, sheet_name="Composite_Data", index=False)

print("Saved correlations_report.xlsx")
