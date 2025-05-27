# sparse_pca_anomaly_detection.py
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import warnings

warnings.filterwarnings("ignore")

FEATURE_DIR = "features/binance_top300"
RESULTS_DIR = "anomaly_detection_results" # 기존 결과와 같은 디렉토리에 저장
os.makedirs(RESULTS_DIR, exist_ok=True)

TOP_K = 10 # Precision@K 계산을 위한 K
MIN_SAMPLES_FOR_AD = 10 # 이상치 탐지를 위한 최소 샘플 수

all_sparse_pca_detection_results = []

print(f"--- Starting Sparse PCA based Anomaly Detection ---")

for filename in os.listdir(FEATURE_DIR):
    if filename.startswith("Z_sparse_pca_") and filename.endswith(".npy"):
        coin = filename[13:-4] # Z_sparse_pca_ 제거, .npy 제거
        z_sparse_pca_path = os.path.join(FEATURE_DIR, filename)
        y_path = os.path.join(FEATURE_DIR, f"y_{coin}.npy")

        if not os.path.exists(y_path):
            print(f"{coin} skipped Sparse PCA AD: Missing y file.")
            continue

        try:
            Z_sparse_pca = np.load(z_sparse_pca_path)
            y_true = np.load(y_path)

            # --- NaN 값 처리 ---
            Z_sparse_pca_finite = np.where(np.isfinite(Z_sparse_pca), Z_sparse_pca, 0)
            
            if len(Z_sparse_pca_finite) < MIN_SAMPLES_FOR_AD or np.sum(y_true) < 1:
                print(f"{coin} skipped Sparse PCA AD: Not enough samples ({len(Z_sparse_pca_finite)}) or no moonshots ({int(np.sum(y_true))}).")
                continue

            print(f"\n--- Processing {coin} for Sparse PCA Anomaly Detection ---")

            # ----------------------------------------------------
            # 1. Sparse PCA 잠재 공간 + LOF 이상치 탐지
            # ----------------------------------------------------
            n_neighbors_lof_sparse = min(20, len(Z_sparse_pca_finite) - 1) 
            sparse_pca_lof_scores = np.full(len(Z_sparse_pca_finite), np.nan) # 기본값
            if n_neighbors_lof_sparse >= 1: # 최소 이웃 수가 1 이상일 때만 LOF 적용
                lof_sparse_model = LocalOutlierFactor(n_neighbors=n_neighbors_lof_sparse)
                sparse_pca_lof_scores = -lof_sparse_model.fit(Z_sparse_pca_finite).negative_outlier_factor_ 
            else:
                print(f"{coin} skipped SparsePCA_LOF: n_neighbors is too small ({n_neighbors_lof_sparse}).")

            # ----------------------------------------------------
            # 2. Sparse PCA 잠재 공간 + Isolation Forest 이상치 탐지
            # ----------------------------------------------------
            if_sparse_model = IsolationForest(random_state=42, n_estimators=100)
            sparse_pca_if_scores = -if_sparse_model.fit(Z_sparse_pca_finite).decision_function(Z_sparse_pca_finite) 

            # --- 각 결과 계산 및 저장 ---
            methods_to_evaluate = {
                "SparsePCA_LOF": sparse_pca_lof_scores,
                "SparsePCA_IF": sparse_pca_if_scores
            }

            for method_name, scores in methods_to_evaluate.items():
                if len(scores) == 0 or np.isnan(scores).all():
                    print(f"Skipping {method_name} for {coin}: No valid scores generated.")
                    continue

                top_indices = np.argsort(scores)[::-1][:TOP_K]
                
                y_pred_binary_at_k = np.zeros_like(y_true)
                if len(top_indices) > 0:
                    y_pred_binary_at_k[top_indices] = 1

                precision_at_k = np.sum(y_true[top_indices]) / TOP_K if TOP_K > 0 else 0.0

                current_recall = np.nan
                current_f1 = np.nan
                current_roc_auc = np.nan
                current_ap = np.nan

                if np.sum(y_true) > 0:
                    current_recall = recall_score(y_true, y_pred_binary_at_k)
                    current_f1 = f1_score(y_true, y_pred_binary_at_k)
                
                if len(np.unique(y_true)) > 1 and not np.isnan(scores).all():
                    try:
                        current_roc_auc = roc_auc_score(y_true, scores)
                    except ValueError:
                        current_roc_auc = np.nan
                    try:
                        current_ap = average_precision_score(y_true, scores)
                    except ValueError:
                        current_ap = np.nan

                print(f"  {method_name} for {coin}: P@{TOP_K}={precision_at_k:.2f}, R={current_recall:.2f}, F1={current_f1:.2f}, ROC_AUC={current_roc_auc:.2f}, AP={current_ap:.2f}")

                all_sparse_pca_detection_results.append({
                    "coin": coin,
                    "num_samples": len(y_true),
                    "num_moonshots": int(np.sum(y_true)),
                    "algorithm": method_name,
                    "Precision@10": round(precision_at_k, 2),
                    "Recall": round(current_recall, 2) if not np.isnan(current_recall) else np.nan,
                    "F1": round(current_f1, 2) if not np.isnan(current_f1) else np.nan,
                    "ROC_AUC": round(current_roc_auc, 2) if not np.isnan(current_roc_auc) else np.nan,
                    "AP": round(current_ap, 2) if not np.isnan(current_ap) else np.nan,
                })

        except Exception as e:
            print(f"Failed to process {coin} for Sparse PCA Anomaly Detection: {e}")

# 결과 저장
df_results = pd.DataFrame(all_sparse_pca_detection_results).drop_duplicates(subset=["coin", "algorithm"])
output_csv_path = os.path.join(RESULTS_DIR, "sparse_pca_anomaly_detection_results.csv")
df_results.to_csv(output_csv_path, index=False)
print(f"\n--- Sparse PCA Anomaly Detection analyses complete. Results saved to {output_csv_path} ---")