import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from pathlib import Path

# ---------- Paths ----------
DOWNLOADS = Path(r"C:\Users\RY163UL\Downloads")
FILES = {
    "demo":   DOWNLOADS / "Demo Analysis.xlsx",
    "tam":    DOWNLOADS / "TAM Analysis.xlsx",
    "tse":    DOWNLOADS / "TSE Analysis.xlsx",
    "soc":    DOWNLOADS / "SOC Analysis.xlsx",
    "trust":  DOWNLOADS / "Trust Analysis.xlsx",
    "env":    DOWNLOADS / "Env Analysis.xlsx",
    "public": DOWNLOADS / "Public Analysis.xlsx",
}
OUT_XLSX = DOWNLOADS / "pca_kmeans_report.xlsx"
FIG1 = DOWNLOADS / "pca_clusters_pc12.png"
FIG2 = DOWNLOADS / "age_vs_intention_clusters.png"

def mean_scale(df, prefix=None, exclude=None):
    if exclude is None: exclude = []
    df = df.copy()
    if prefix is not None:
        cols = [c for c in df.columns if str(c).startswith(prefix)]
    else:
        cols = [c for c in df.columns if c not in (["Participant No"] + exclude)]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[cols].mean(axis=1, skipna=True)

# ---------- Load & clean ----------
dfs = {k: pd.read_excel(p).rename(columns=lambda c: str(c).strip()) for k, p in FILES.items()}
for k in dfs:
    if "Participant No" in dfs[k].columns:
        dfs[k] = dfs[k][dfs[k]["Participant No"] != 86].copy()  # remove participant 86

parts = dfs["tam"][["Participant No"]].drop_duplicates().rename(columns={"Participant No": "participant"})

tam = dfs["tam"].copy()
tam["PU_mean"]   = mean_scale(tam, prefix="PU")
tam["PEOU_mean"] = mean_scale(tam, prefix="PEOU")
tam["IU_mean"]   = mean_scale(tam, prefix="IU")

dfs["tse"]["TSE_mean"]       = mean_scale(dfs["tse"])
dfs["soc"]["SOC_mean"]       = mean_scale(dfs["soc"])
dfs["trust"]["Trust_mean"]   = mean_scale(dfs["trust"])
dfs["env"]["Env_mean"]       = mean_scale(dfs["env"])
dfs["public"]["Public_mean"] = mean_scale(dfs["public"])

def pick(df, cols): return df[["Participant No"] + cols].copy()

merged = parts.copy()
merged = merged.merge(
    pick(tam, ["PU_mean", "PEOU_mean", "IU_mean"]),
    left_on="participant", right_on="Participant No", how="left"
).drop(columns=["Participant No"])

for nm, cols in [
    ("tse", ["TSE_mean"]), ("soc", ["SOC_mean"]), ("trust", ["Trust_mean"]),
    ("env", ["Env_mean"]), ("public", ["Public_mean"])
]:
    merged = merged.merge(
        pick(dfs[nm], cols),
        left_on="participant", right_on="Participant No", how="left"
    ).drop(columns=["Participant No"])

demo = dfs["demo"].copy()
age_col = next((c for c in demo.columns if "age" in c.lower()), None)
if age_col is None:
    raise ValueError("Age column not found")
merged = merged.merge(
    demo[["Participant No", age_col]].rename(columns={age_col: "age"}),
    left_on="participant", right_on="Participant No", how="left"
).drop(columns=["Participant No"])
merged["age"] = pd.to_numeric(merged["age"], errors="coerce")

df = merged.dropna(subset=["age", "IU_mean"]).copy()

# ---------- Standardize + age weighting ----------
features = ["age", "PU_mean", "PEOU_mean", "IU_mean", "TSE_mean", "SOC_mean", "Trust_mean", "Env_mean", "Public_mean"]
X = df[features].values
scaler = StandardScaler()
Xz = scaler.fit_transform(X)
Xz[:, 0] *= 2.0  # age is primary

# ---------- Choose k via silhouette ----------
sil, models = {}, {}
for k in (2, 3, 4, 5, 6):
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labs = km.fit_predict(Xz)
    sil[k] = silhouette_score(Xz, labs); models[k] = km
best_k = max(sil, key=lambda k: (sil[k], -k))
km = models[best_k]
df["cluster"] = km.predict(Xz)

# ---------- PCA (visualisation only) ----------
pca = PCA(n_components=2, random_state=42)
Z2 = pca.fit_transform(Xz)                  # PC scores
centroids_pc = pca.transform(km.cluster_centers_)

# Also report centroids in original measurement scale
cent_z = km.cluster_centers_.copy()
cent_z[:, 0] /= 2.0
cent_orig = scaler.inverse_transform(cent_z)
centroids_orig_df = pd.DataFrame(cent_orig, columns=features)
centroids_orig_df.insert(0, "cluster", range(len(centroids_orig_df)))

# ---------- Figures ----------
# PCA1 vs PCA2 with centroids
ev = pca.explained_variance_ratio_
plt.figure()
for c in sorted(df["cluster"].unique()):
    pts = Z2[df["cluster"].values == c]
    plt.scatter(pts[:, 0], pts[:, 1], label=f"Cluster {c}", alpha=0.7, marker='x')
plt.scatter(centroids_pc[:, 0], centroids_pc[:, 1], marker='X', s=200, label="Centroid")
plt.title(f"PCA1 vs PCA2 (k={best_k})")
plt.xlabel(f"PC1 ({ev[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({ev[1]*100:.1f}% var)")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(FIG1, dpi=200)
plt.show()

# Age vs Intention with centroids
plt.figure()
for c in sorted(df["cluster"].unique()):
    sub = df[df["cluster"] == c]
    plt.scatter(sub["age"], sub["IU_mean"], label=f"Cluster {c}", alpha=0.7, marker='x')
for i in range(centroids_orig_df.shape[0]):
    plt.scatter(centroids_orig_df.loc[i, "age"], centroids_orig_df.loc[i, "IU_mean"], marker='X', s=200, label=f"Centroid {i}")
plt.title(f"Age vs Adoption Intention by cluster (k={best_k})")
plt.xlabel("Age (years)"); plt.ylabel("Adoption Intention (1–5)")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(FIG2, dpi=200)
plt.show()

# ---------- Save compact report ----------
pca_loadings = pd.DataFrame(pca.components_.T, index=features, columns=["PC1_loading", "PC2_loading"])
with pd.ExcelWriter(OUT_XLSX, engine="xlsxwriter") as w:
    pd.DataFrame({"k": list(sil.keys()), "silhouette": [sil[k] for k in sil]}).to_excel(
        w, index=False, sheet_name="Model_Selection"
    )
    pd.DataFrame(Z2, columns=["PC1", "PC2"]).assign(cluster=df["cluster"].values).to_excel(
        w, index=False, sheet_name="Scores_PC12"
    )
    pca_loadings.to_excel(w, sheet_name="Loadings_PC12")
    centroids_orig_df.to_excel(w, index=False, sheet_name="Centroids_OrigScale")

print(f"[Saved] {OUT_XLSX}")
print(f"[Saved] {FIG1}")
print(f"[Saved] {FIG2}")
