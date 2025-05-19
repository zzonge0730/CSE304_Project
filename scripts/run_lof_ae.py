import os
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

FEATURE_DIR = "features/binance_top300"
TOP_K = 10
results = []
seen = set()

for file in os.listdir(FEATURE_DIR):
    if file.startswith("Z_ae_") and file.endswith(".npy"):
        coin = file[7:-4]
        if coin in seen:
            continue
        seen.add(coin)

        z_path = os.path.join(FEATURE_DIR, f"Z_ae_{coin}.npy")
        y_path = os.path.join(FEATURE_DIR, f"y_{coin}.npy")
        if not os.path.exists(y_path):
            continue

        Z = np.load(z_path)
        y = np.load(y_path)

        if len(Z) < 5 or np.sum(y) < 1:
            print(f"⚠️ {coin} skipped: {len(Z)} samples, {int(np.sum(y))} moonshots")
            continue

        try:
            n_neighbors = min(20, len(Z) - 1)
            lof = LocalOutlierFactor(n_neighbors=n_neighbors)
            scores = -lof.fit(Z).negative_outlier_factor_
            top_indices = np.argsort(scores)[::-1][:TOP_K]
            precision = np.sum(y[top_indices]) / TOP_K

            print(f"💡 {coin}: AE+LOF top-{TOP_K} precision = {precision:.2f} ({int(np.sum(y[top_indices]))}/{TOP_K})")

            results.append({
                "coin": coin,
                "num_samples": len(y),
                "num_moonshots": int(np.sum(y)),
                "AE_LOF_Precision@10": round(precision, 2)
            })
        except Exception as e:
            print(f"❌ {coin} failed: {e}")

# 저장
df = pd.DataFrame(results).drop_duplicates(subset=["coin"])
df.to_csv("ae_lof_results.csv", index=False)
print("📁 Saved: ae_lof_results.csv")
