# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 14:13:03 2025

@author: RY163UL
"""


import numpy as np
import pandas as pd
from numpy.linalg import norm

# -------------------
# 0) Paths (Windows local)
# -------------------
PATHS = {
    "Public": r"C:\Users\RY163UL\Downloads\Public Analysis.xlsx",
    "ENV":    r"C:\Users\RY163UL\Downloads\Env Analysis.xlsx",
    "Trust":  r"C:\Users\RY163UL\Downloads\Trust Analysis.xlsx",
    "TSE":    r"C:\Users\RY163UL\Downloads\TSE Analysis.xlsx",
    "SOC":    r"C:\Users\RY163UL\Downloads\SOC Analysis.xlsx",
    "TAM":    r"C:\Users\RY163UL\Downloads\TAM Analysis.xlsx",
}

# -------------------
# 1) Load and harmonise
# -------------------
dfs = {k: pd.read_excel(v).rename(columns=lambda c: str(c).strip()) for k, v in PATHS.items()}

# Remove Participant 86; keep a consistent participant universe from TAM
for k in dfs:
    if "Participant No" in dfs[k].columns:
        dfs[k] = dfs[k][dfs[k]["Participant No"] != 86].copy()

participants = (
    dfs["TAM"][["Participant No"]]
    .drop_duplicates()
    .rename(columns={"Participant No": "participant"})
)

def _make_block(df, item_cols):
    """Align to participants, numeric cast, mean-impute per item, z-score per item."""
    out = df[["Participant No"] + item_cols].rename(columns={"Participant No": "participant"})
    out = participants.merge(out, on="participant", how="left")
    for c in out.columns:
        if c == "participant":
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out[c] = out[c].fillna(out[c].mean())
        # z-score with sample std
        out[c] = (out[c] - out[c].mean()) / out[c].std(ddof=1)
    return out

blocks = {
    "PU":   _make_block(dfs["TAM"],  [c for c in dfs["TAM"].columns if str(c).startswith("PU")]),
    "PEOU": _make_block(dfs["TAM"],  [c for c in dfs["TAM"].columns if str(c).startswith("PEOU")]),
    "Adoption Intention": _make_block(dfs["TAM"], [c for c in dfs["TAM"].columns if str(c).startswith("IU")]),
    "TSE":  _make_block(dfs["TSE"],  [c for c in dfs["TSE"].columns if c != "Participant No"]),
    "SOC":  _make_block(dfs["SOC"],  [c for c in dfs["SOC"].columns if c != "Participant No"]),
    "Trust":_make_block(dfs["Trust"],[c for c in dfs["Trust"].columns if c != "Participant No"]),
    "Environment": _make_block(dfs["ENV"], [c for c in dfs["ENV"].columns if c != "Participant No"]),
    "Public": _make_block(dfs["Public"], [c for c in dfs["Public"].columns if c != "Participant No"]),
}
lvs = list(blocks.keys())
X = {lv: blocks[lv].drop(columns=["participant"]).values for lv in lvs}
cols_map = {lv: [c for c in blocks[lv].columns if c != "participant"] for lv in lvs}
N = list(X.values())[0].shape[0]

# -------------------
# 2) Core PLS algorithm (Mode A, path weighting)
# -------------------
def plspm(X_dict, edges, max_iter=300, tol=1e-6):
    """
    Mode A, path weighting.
    edges: set of tuples (a, b) meaning a--b adjacency for inner proxies (undirected for proxy stage).
    """
    # neighbors for inner approximation
    neighbors = {lv: set() for lv in X_dict}
    for a, b in edges:
        neighbors[a].add(b)
        neighbors[b].add(a)

    # init outer weights equally
    W = {lv: np.ones(X_dict[lv].shape[1]) / X_dict[lv].shape[1] for lv in X_dict}
    last = {lv: W[lv].copy() for lv in X_dict}

    for _ in range(max_iter):
        # latent scores and standardise
        Y = {lv: X_dict[lv] @ W[lv] for lv in X_dict}
        Y = {lv: (y - y.mean()) / y.std(ddof=1) for lv, y in Y.items()}

        # inner proxies
        Z = {}
        for lv in X_dict:
            if not neighbors[lv]:
                Z[lv] = Y[lv]
            else:
                acc = np.zeros_like(Y[lv])
                for nb in neighbors[lv]:
                    r = np.corrcoef(Y[lv], Y[nb])[0, 1]
                    s = 1.0 if np.isnan(r) or r == 0 else np.sign(r)
                    acc += s * Y[nb]
                Z[lv] = acc

        # Mode A outer weights: corr(indicator, Z_lv)
        for lv in X_dict:
            z = Z[lv]
            w = np.array([np.corrcoef(X_dict[lv][:, j], z)[0, 1] for j in range(X_dict[lv].shape[1])])
            w = np.nan_to_num(w, nan=0.0)
            W[lv] = w / (norm(w) if norm(w) > 0 else 1.0)

        # convergence
        if max(norm(W[lv] - last[lv]) for lv in X_dict) < tol:
            break
        last = {lv: W[lv].copy() for lv in X_dict}

    # final latent scores and loadings
    Y = {lv: (X_dict[lv] @ W[lv] - (X_dict[lv] @ W[lv]).mean()) / (X_dict[lv] @ W[lv]).std(ddof=1) for lv in X_dict}
    L = {lv: np.array([np.corrcoef(X_dict[lv][:, j], Y[lv])[0, 1] for j in range(X_dict[lv].shape[1])]) for lv in X_dict}
    return W, Y, L

# Sign-correct indicators (so outer loadings are positive)
W0, Y0, L0 = plspm(X, edges=set())
for lv in lvs:
    for j, ld in enumerate(L0[lv]):
        if ld < 0:
            X[lv][:, j] *= -1

# -------------------
# 3) Utility functions
# -------------------
def regress(endog, preds, Y):
    """OLS on latent scores; returns (coef dict, R²)."""
    y = Y[endog]
    if len(preds) == 0:  # intercept-only
        yhat = np.full_like(y, y.mean())
        r2 = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        return {}, float(r2)
    Xmat = np.column_stack([Y[p] for p in preds])
    X_ = np.column_stack([np.ones(len(y)), Xmat])
    b = np.linalg.lstsq(X_, y, rcond=None)[0]
    yhat = X_ @ b
    r2 = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return dict(zip(preds, b[1:])), float(r2)

def ave_rho(loads):
    """AVE and composite reliability rhoC for reflective blocks."""
    lam2 = loads ** 2
    ave = float(lam2.mean())
    num = float((loads.sum()) ** 2)
    den = num + float(np.sum(1 - lam2))
    rhoC = num / den if den > 0 else np.nan
    return ave, rhoC

def summarise_boot(arr):
    arr = np.array(arr, float)
    m = float(arr.mean()); se = float(arr.std(ddof=1))
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return m, se, float(lo), float(hi)

def t_and_p_from_boot(arr):
    """t = mean/SE; p from normal approx (two-tailed)."""
    m, se, _, _ = summarise_boot(arr)
    t = m / (se + 1e-12)
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(t) / np.sqrt(2))))
    return float(t), float(p)

# -------------------
# 4) Model A (Extended model with Trust/Public/Environment → PU and Trust → PEOU)
# -------------------
edges_A = {
    ("TSE", "PEOU"), ("Trust", "PEOU"),
    ("PEOU", "PU"), ("Trust", "PU"), ("Public", "PU"), ("Environment", "PU"),
    ("TSE", "SOC"),
    ("PU", "Adoption Intention"), ("PEOU", "Adoption Intention"),
    ("TSE", "Adoption Intention"), ("SOC", "Adoption Intention"),
    ("Trust", "Adoption Intention"), ("Environment", "Adoption Intention"), ("Public", "Adoption Intention"),
}
W_A, Y_A, L_A = plspm(X, edges_A)

# Measurement model (AVE, rhoC)
MM_rows = []
for lv in lvs:
    ave, rho = ave_rho(L_A[lv])
    MM_rows.append([lv, len(L_A[lv]), round(ave, 3), round(rho, 3)])
MM = pd.DataFrame(MM_rows, columns=["Construct", "k", "AVE", "rhoC"])

# Structural equations for Model A
endog_A = {
    "PEOU": ["TSE", "Trust"],
    "PU": ["PEOU", "Trust", "Public", "Environment"],
    "SOC": ["TSE"],
    "Adoption Intention": ["PU", "PEOU", "TSE", "SOC", "Trust", "Environment", "Public"],
}
coefs_A, r2_A = {}, {}
for y, preds in endog_A.items():
    c, r2 = regress(y, preds, Y_A)
    coefs_A[y], r2_A[y] = c, r2

# Bootstrap Model A
B_A = 400
rng = np.random.default_rng(123)
boot_paths_A = {y: {p: [] for p in preds} for y, preds in endog_A.items()}
boot_r2_A = {y: [] for y in endog_A}

for b in range(B_A):
    idx = rng.integers(0, N, N)
    Xb = {lv: X[lv][idx, :] for lv in lvs}
    Wb, Yb, Lb = plspm(Xb, edges_A)
    for y, preds in endog_A.items():
        cb, r2b = regress(y, preds, Yb)
        boot_r2_A[y].append(r2b)
        for p in preds:
            boot_paths_A[y][p].append(cb.get(p, np.nan))

# Structural table with 95% CIs
rowsA = []
for y, preds in endog_A.items():
    for p in preds:
        m, se, lo, hi = summarise_boot(boot_paths_A[y][p])
        rowsA.append([y, p, round(coefs_A[y][p], 3), round(m, 3), f"[{lo:.3f}, {hi:.3f}]"])
STRUCT_A = pd.DataFrame(rowsA, columns=["Endogenous", "Predictor", "β (orig)", "β (boot mean)", "95% CI"])

R2_A = pd.DataFrame([
    {"Construct": y,
     "R² (orig)": round(r2_A[y], 3),
     "R² (boot mean)": round(float(np.mean(boot_r2_A[y])), 3),
     "95% CI": f"[{np.percentile(boot_r2_A[y], 2.5):.3f}, {np.percentile(boot_r2_A[y], 97.5):.3f}]"}
    for y in endog_A
])

# Key indirect effects (Model A)
def prod(a, b): return np.array(a) * np.array(b)

IND_rows = []
IND_rows.append(["PEOU → PU → Adoption Intention",
                 *summarise_boot(prod(boot_paths_A["PU"]["PEOU"], boot_paths_A["Adoption Intention"]["PU"]))])
IND_rows.append(["Trust → PU → Adoption Intention",
                 *summarise_boot(prod(boot_paths_A["PU"]["Trust"], boot_paths_A["Adoption Intention"]["PU"]))])
IND_rows.append(["Trust → PEOU → Adoption Intention",
                 *summarise_boot(prod(boot_paths_A["PEOU"]["Trust"], boot_paths_A["Adoption Intention"]["PEOU"]))])
IND_rows.append(["Trust → PEOU → PU → Adoption Intention",
                 *summarise_boot(prod(prod(boot_paths_A["PEOU"]["Trust"], boot_paths_A["PU"]["PEOU"]),
                                      boot_paths_A["Adoption Intention"]["PU"]))])
IND_rows.append(["TSE → PEOU → Adoption Intention",
                 *summarise_boot(prod(boot_paths_A["PEOU"]["TSE"], boot_paths_A["Adoption Intention"]["PEOU"]))])
IND_rows.append(["TSE → PEOU → PU → Adoption Intention",
                 *summarise_boot(prod(prod(boot_paths_A["PEOU"]["TSE"], boot_paths_A["PU"]["PEOU"]),
                                      boot_paths_A["Adoption Intention"]["PU"]))])
IND_rows.append(["TSE → SOC → Adoption Intention",
                 *summarise_boot(prod(boot_paths_A["SOC"]["TSE"], boot_paths_A["Adoption Intention"]["SOC"]))])
IND_rows.append(["Public → PU → Adoption Intention",
                 *summarise_boot(prod(boot_paths_A["PU"]["Public"], boot_paths_A["Adoption Intention"]["PU"]))])
IND_rows.append(["Environment → PU → Adoption Intention",
                 *summarise_boot(prod(boot_paths_A["PU"]["Environment"], boot_paths_A["Adoption Intention"]["PU"]))])

IND_A = pd.DataFrame(IND_rows, columns=["Indirect effect", "Boot mean", "SE", "2.5%", "97.5%"])
IND_A["95% CI"] = IND_A.apply(lambda r: f"[{r['2.5%']:.3f}, {r['97.5%']:.3f}]", axis=1)
IND_A = IND_A.drop(columns=["2.5%", "97.5%"])
IND_A["Boot mean"] = IND_A["Boot mean"].round(3)
IND_A["SE"] = IND_A["SE"].round(3)

# -------------------
# 5) Model B (User’s hypotheses table: β, t, p, f², Decision)
# -------------------
edges_B = {
    ("PEOU", "TSE"), ("PU", "TSE"),
    ("TSE", "SOC"),
    ("PU", "Adoption Intention"), ("PEOU", "Adoption Intention"), ("Trust", "Adoption Intention"),
    ("Environment", "Adoption Intention"), ("Public", "Adoption Intention"),
    ("TSE", "Adoption Intention"), ("SOC", "Adoption Intention"),
}
W_B, Y_B, L_B = plspm(X, edges_B)

endog_B = {
    "TSE": ["PEOU", "PU"],
    "SOC": ["TSE"],
    "Adoption Intention": ["PU", "PEOU", "Trust", "Environment", "Public", "TSE", "SOC"],
}
coefs_B, r2_B = {}, {}
for y, preds in endog_B.items():
    c, r2 = regress(y, preds, Y_B)
    coefs_B[y], r2_B[y] = c, r2

def f2_for_pred(y, pred, preds, Ylatent):
    r2_full = r2_B[y]
    preds_reduced = [q for q in preds if q != pred]
    _, r2_reduced = regress(y, preds_reduced, Ylatent)
    return (r2_full - r2_reduced) / (1 - r2_full + 1e-12)

# compute f² values
f2_B = {(y, p): f2_for_pred(y, p, preds, Y_B) for y, preds in endog_B.items() for p in preds}

# bootstrap for t and p (user table)
B_B = 600
rng = np.random.default_rng(456)
boot_paths_B = {(y, p): [] for y, preds in endog_B.items() for p in preds}

for b in range(B_B):
    idx = rng.integers(0, N, N)
    Xb = {lv: X[lv][idx, :] for lv in lvs}
    Wb, Yb, Lb = plspm(Xb, edges_B)
    for y, preds in endog_B.items():
        cb, _ = regress(y, preds, Yb)
        for p in preds:
            boot_paths_B[(y, p)].append(cb.get(p, np.nan))

def t_and_p(arr):
    t, p = t_and_p_from_boot(arr)
    return round(float(t), 2), ("<.001" if p < 0.001 else f"{p:.3f}")

order = [
    ("TSE", "PEOU"), ("TSE", "PU"), ("SOC", "TSE"),
    ("Adoption Intention", "PU"), ("Adoption Intention", "PEOU"), ("Adoption Intention", "Trust"),
    ("Adoption Intention", "Environment"), ("Adoption Intention", "Public"),
    ("Adoption Intention", "TSE"), ("Adoption Intention", "SOC"),
]

rowsB = []
for y, p in order:
    beta = round(coefs_B[y][p], 3)
    tval, pval = t_and_p(boot_paths_B[(y, p)])
    f2v = round(f2_B[(y, p)], 2)
    decision = "Supported" if (pval == "<.001" or float(pval) < 0.05) else "Not supported"
    rowsB.append([f"{y} ← {p}", beta, tval, pval, f2v, decision])

TAB_B = pd.DataFrame(rowsB, columns=["Endogenous → Predictor", "β", "t-value", "p", "f²", "Decision"])

# -------------------
# 6) Save all tables to one Excel workbook (Downloads folder)
# -------------------
OUT = r"C:\Users\RY163UL\Downloads\pls_sem_full_tables.xlsx"
with pd.ExcelWriter(OUT, engine="xlsxwriter") as w:
    MM.to_excel(w, index=False, sheet_name="Measurement_Model")
    STRUCT_A.to_excel(w, index=False, sheet_name="Structural_Paths_CI")
    R2_A.to_excel(w, index=False, sheet_name="R2_Extended")
    IND_A.to_excel(w, index=False, sheet_name="Indirect_Effects")
    TAB_B.to_excel(w, index=False, sheet_name="Hypotheses_Table")

print(f"Saved: {OUT}")
