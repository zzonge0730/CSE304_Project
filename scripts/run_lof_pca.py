# run_lof_pca.py
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import recall_score, f1_score, roc_auc_score, average_precision_score

FEATURE_DIR = "features/binance_top300"
TOP_K = 10
results = []

for file in os.listdir(FEATURE_DIR):
    if not file.startswith("Z_pca_") or not file.endswith(".npy"):
        continue

    coin = file[6:-4]
    z_path = os.path.join(FEATURE_DIR, f"Z_pca_{coin}.npy")
    y_path = os.path.join(FEATURE_DIR, f"y_{coin}.npy")
    if not os.path.exists(y_path):
        continue

    Z = np.load(z_path)
    y = np.load(y_path)

    if len(Z) < 5 or np.sum(y) < 1:
        continue

    try:
        lof = LocalOutlierFactor(n_neighbors=min(20, len(Z) - 1))
        scores = -lof.fit(Z).negative_outlier_factor_
        top_indices = np.argsort(scores)[::-1][:TOP_K]
        precision = np.sum(y[top_indices]) / TOP_K

        print(f"{coin}: PCA+LOF top-{TOP_K} precision = {precision:.2f}")

        # 1. Top-K 기준 이진 예측
        y_pred = np.zeros_like(y)
        y_pred[top_indices] = 1

        # 2. 지표 계산
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        roc_auc = roc_auc_score(y, scores)
        ap = average_precision_score(y, scores)        
        print(f"{coin}: AE+IForest top-{TOP_K} precision = {precision:.2f}")

        results.append({
            "coin": coin,
            "num_samples": len(y),
            "num_moonshots": int(np.sum(y)),
            "LOF_PCA_Precision@10": round(precision, 2),
            "Recall": round(recall, 2),
            "F1": round(f1, 2),
            "ROC_AUC": round(roc_auc, 3),
            "AP": round(ap, 3)
        })
    except Exception as e:
        print(f"{coin} failed: {e}")

df = pd.DataFrame(results)
df.to_csv("lof_pca_results.csv", index=False)
print("Saved: lof_pca_results.csv")
