# quick_expansion.py   ←  반드시 umap.py 가 아닌 다른 이름!
from pathlib import Path
import numpy as np
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, roc_auc_score
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler   
import pandas as pd

# ---------- 0. load & stack ----------
feat_dir  = Path("features/binance_top300")

X_list, y_list = [], []

for xfile in sorted(feat_dir.glob("X_*.npy")):
    coin  = xfile.stem[2:]                 # BTCUSDT
    yfile = feat_dir / f"y_{coin}.npy"    
    if not yfile.exists():
        continue
    X_list.append(np.load(xfile))
    y_list.append(np.load(yfile))

X_raw = np.vstack(X_list)                  # (N, 72)
y_all = np.hstack(y_list)                  # (N,)
print("Loaded", X_raw.shape, "labels:", y_all.sum())

# ---------- 0.1  NaN / Inf filter ----------
mask = np.isfinite(X_raw).all(axis=1)
if not mask.all():
    print("Removing rows with NaN/Inf:", (~mask).sum())
    X_raw = X_raw[mask]
    y_all = y_all[mask]
scaler = StandardScaler()
X_raw = scaler.fit_transform(X_raw) 

# ---------- 1. UMAP + KMeans ----------
X_umap = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0)\
            .fit_transform(X_raw)
sil_umap = silhouette_score(
    X_umap,
    KMeans(3, n_init=10, random_state=0).fit_predict(X_umap)
)
print(f"UMAP (2-D) Silhouette = {sil_umap:.3f}")

# ---------- 2. AE latent sweep ----------
def build_ae(latent):
    inp = layers.Input(shape=(72,))
    x   = layers.Dense(64, activation='relu')(inp)
    x   = layers.Dense(32, activation='relu')(x)
    lat = layers.Dense(latent, activation='relu')(x)
    x   = layers.Dense(32, activation='relu')(lat)
    x   = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(72)(x)
    return models.Model(inp, out), models.Model(inp, lat)

rows = []
for k in [4, 8, 16]:
    ae, enc = build_ae(k)
    ae.compile('adam', 'mse')
    ae.fit(X_raw, X_raw, epochs=10, batch_size=256, verbose=0)

    Z   = enc.predict(X_raw, verbose=0)
    if np.isnan(Z).any():
        print(f"NaN detected in Z (k={k}); replacing with 0")
        Z = np.nan_to_num(Z, nan=0.0)

    sil = silhouette_score(Z, KMeans(3, n_init=10, random_state=0)
                                  .fit_predict(Z))

    recon = np.square(X_raw - ae.predict(X_raw, verbose=0)).mean(axis=1)
    recon = np.nan_to_num(recon, nan=0.0)     # ← 재구성 오류 NaN 대체
    roc   = roc_auc_score(y_all, recon)
    sil = silhouette_score(Z, KMeans(3, n_init=10, random_state=0)\
                                  .fit_predict(Z))
    recon = np.mean(np.square(X_raw - ae.predict(X_raw, verbose=0)), axis=1)
    roc  = roc_auc_score(y_all, recon)
    rows.append((k, sil, roc))
    print(f"AE-{k:>2}D  Silhouette={sil:.3f}  ROC-AUC={roc:.3f}")

# ---------- 3. save as CSV / markdown ----------
df = pd.DataFrame(rows, columns=["Dim", "Silhouette", "ROC_AUC"])
df.to_csv("ae_sweep_results.csv", index=False)
print("\n", df.to_markdown(index=False))
