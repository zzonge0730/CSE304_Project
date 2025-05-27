# run_iforest_binance.py
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import recall_score, f1_score, roc_auc_score, average_precision_score

FEATURE_DIR = "features/binance_top300"
TOP_K = 10
results = []

for file in os.listdir(FEATURE_DIR):
    if file.startswith("X_") and file.endswith(".npy"):
        coin = file[2:-4]
        x_path = os.path.join(FEATURE_DIR, f"X_{coin}.npy")
        y_path = os.path.join(FEATURE_DIR, f"y_{coin}.npy")
        if not os.path.exists(y_path):
            continue

        X = np.load(x_path)
        y = np.load(y_path)

        if len(y) < 10 or np.sum(y) < 2:
            continue

        iforest = IsolationForest(random_state=42)
        scores = -iforest.fit(X).score_samples(X)
        top_indices = np.argsort(scores)[::-1][:TOP_K]
        precision = np.sum(y[top_indices]) / TOP_K
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
            "IF_Precision@10": round(precision, 2),
            "Recall": round(recall, 2),
            "F1": round(f1, 2),
            "ROC_AUC": round(roc_auc, 3),
            "AP": round(ap, 3)
        })
# 결과 저장
df = pd.DataFrame(results)
df.to_csv("iforest_results.csv", index=False)
print("Saved: iforest_results.csv")
