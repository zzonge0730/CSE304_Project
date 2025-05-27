# compress_pca_multi.py
import os
import numpy as np
from sklearn.decomposition import PCA

FEATURE_DIR = "features/binance_top300"
OUTPUT_PREFIX = "Z_pca"
TARGET_DIM = 10  # 원하는 PCA 차원 수

for file in os.listdir(FEATURE_DIR):
    if not file.startswith("X_") or not file.endswith(".npy"):
        continue

    coin = file[2:-4]
    x_path = os.path.join(FEATURE_DIR, f"X_{coin}.npy")
    y_path = os.path.join(FEATURE_DIR, f"y_{coin}.npy")

    if not os.path.exists(y_path):
        continue

    try:
        X = np.load(x_path)
        y = np.load(y_path)

        if len(X) < 5 or np.sum(y) < 1:
            print(f"⚠️ {coin}: skipped due to small sample size")
            continue

        pca = PCA(n_components=TARGET_DIM)
        Z = pca.fit_transform(X)

        out_path = os.path.join(FEATURE_DIR, f"{OUTPUT_PREFIX}_{coin}.npy")
        np.save(out_path, Z)

        print(f"PCA compressed {coin}: {X.shape} → {Z.shape}")

    except Exception as e:
        print(f"{coin} failed: {e}")
