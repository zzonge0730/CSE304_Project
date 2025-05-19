# compress_and_detect.py
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

FEATURE_DIR = "features/top50"
RESULT_PATH = "results.csv"

results = []

for file in os.listdir(FEATURE_DIR):
    if not file.startswith("X_") or not file.endswith(".npy"):
        continue

    coin = file[2:-4]  # strip 'X_' and '.npy'
    X = np.load(os.path.join(FEATURE_DIR, file))
    y_path = os.path.join(FEATURE_DIR, f"y_{coin}.npy")
    if not os.path.exists(y_path):
        continue
    y = np.load(y_path)

    if sum(y) < 3 or len(y) < 10:
        print(f"⚠️ Skipping {coin}: insufficient labels")
        continue

    # PCA 압축
    pca = PCA(n_components=2)
    Z_pca = pca.fit_transform(X)

    # LOF
    lof = LocalOutlierFactor(n_neighbors=20)
    lof_score = -lof.fit(Z_pca).negative_outlier_factor_
    lof_rank = np.argsort(lof_score)[::-1]
    lof_precision = sum(y[lof_rank[:10]]) / 10

    # Isolation Forest
    iforest = IsolationForest(n_estimators=100)
    iforest_score = -iforest.fit(X).score_samples(X)
    iforest_rank = np.argsort(iforest_score)[::-1]
    iforest_precision = sum(y[iforest_rank[:10]]) / 10

    results.append({
        "coin": coin,
        "num_samples": len(y),
        "num_moonshots": int(sum(y)),
        "LOF_PCA_Precision@10": round(lof_precision, 2),
        "IF_Original_Precision@10": round(iforest_precision, 2)
    })

# 결과 저장
pd.DataFrame(results).to_csv(RESULT_PATH, index=False)
print(f"✅ Results saved to {RESULT_PATH}")
