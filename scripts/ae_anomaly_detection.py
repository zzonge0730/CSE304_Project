# ae_anomaly_detection.py (수정 제안 부분)

import os
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import warnings

warnings.filterwarnings("ignore")

FEATURE_DIR = "features/binance_top300"
RESULTS_DIR = "anomaly_detection_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

TOP_K = 10
LATENT_DIM = 8

def build_autoencoder_for_reconstruction(input_dim, latent_dim):
    inp = Input(shape=(input_dim,))
    encoded = Dense(32, activation='relu')(inp)
    encoded = Dense(latent_dim, activation='relu')(encoded)
    decoded = Dense(32, activation='relu')(encoded)
    out = Dense(input_dim, activation='linear')(decoded)
    model = Model(inputs=inp, outputs=out)
    return model

all_ae_detection_results = []

for file in os.listdir(FEATURE_DIR):
    if file.startswith("X_") and file.endswith(".npy"):
        coin = file[2:-4]
        x_path = os.path.join(FEATURE_DIR, f"X_{coin}.npy")
        y_path = os.path.join(FEATURE_DIR, f"y_{coin}.npy")
        z_ae_path = os.path.join(FEATURE_DIR, f"Z_ae_{coin}.npy")

        if not os.path.exists(y_path) or not os.path.exists(z_ae_path):
            print(f"⚠️ {coin} skipped AE AD: Missing y or Z_ae file.")
            continue

        try:
            X_raw = np.load(x_path)
            Z_ae = np.load(z_ae_path)
            y_true = np.load(y_path)

            # --- NaN 값 처리 추가 ---
            # 1. X_raw에서 NaN/inf 처리
            # np.isfinite는 유한한(finite) 값인지 확인. NaN, inf, -inf는 False
            X_raw_finite = np.where(np.isfinite(X_raw), X_raw, 0) # NaN/inf를 0으로 대체
            # 또는 X_raw_finite = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0) # 더 명시적
            
            # 2. Z_ae에서 NaN/inf 처리 (AutoEncoder 결과도 NaN을 포함할 수 있음)
            Z_ae_finite = np.where(np.isfinite(Z_ae), Z_ae, 0) # NaN/inf를 0으로 대체
            # 또는 Z_ae_finite = np.nan_to_num(Z_ae, nan=0.0, posinf=0.0, neginf=0.0)

            # 처리 후 다시 샘플 수 및 문샷 존재 확인
            if len(X_raw_finite) < 10 or np.sum(y_true) < 1:
                print(f"⚠️ {coin} skipped AE AD: Not enough samples ({len(X_raw_finite)}) or no moonshots ({int(np.sum(y_true))}).")
                continue

            print(f"\n--- Processing {coin} for AE Anomaly Detection ---")

            # ----------------------------------------------------
            # 1. AE 재구성 오차 기반 이상치 탐지 (X_raw_finite 사용)
            # ----------------------------------------------------
            ae_model = build_autoencoder_for_reconstruction(X_raw_finite.shape[1], LATENT_DIM)
            ae_model.compile(optimizer='adam', loss='mse')
            ae_model.fit(X_raw_finite, X_raw_finite, epochs=50, batch_size=16, verbose=0)
            
            reconstructions = ae_model.predict(X_raw_finite, verbose=0)
            reconstruction_errors = np.mean(np.square(X_raw_finite - reconstructions), axis=1)
            ae_reconstruction_scores = reconstruction_errors

            # ----------------------------------------------------
            # 2. AE 잠재 공간 (Z_ae_finite) + LOF 이상치 탐지
            # ----------------------------------------------------
            n_neighbors_lof = min(20, len(Z_ae_finite) - 1) 
            if n_neighbors_lof < 1: # 최소 이웃 수가 1 미만이면 LOF 적용 불가
                print(f"⚠️ {coin} skipped AE_LOF: n_neighbors is too small ({n_neighbors_lof}).")
                ae_lof_scores = np.full(len(Z_ae_finite), np.nan) # 점수 NaN으로 채우기
            else:
                lof_model = LocalOutlierFactor(n_neighbors=n_neighbors_lof)
                ae_lof_scores = -lof_model.fit(Z_ae_finite).negative_outlier_factor_ 

            # ----------------------------------------------------
            # 3. AE 잠재 공간 (Z_ae_finite) + Isolation Forest 이상치 탐지
            # ----------------------------------------------------
            if_model = IsolationForest(random_state=42, n_estimators=100)
            ae_if_scores = -if_model.fit(Z_ae_finite).decision_function(Z_ae_finite) 

            # ... (이하 동일) ...
            methods_to_evaluate = {
                "AE_Reconstruction": ae_reconstruction_scores,
                "AE_LOF": ae_lof_scores,
                "AE_IF": ae_if_scores
            }

            for method_name, scores in methods_to_evaluate.items():
                if len(scores) == 0 or np.isnan(scores).all(): # 점수 자체가 모두 NaN이면 스킵
                    print(f"Skipping {method_name} for {coin}: No valid scores generated.")
                    continue

                top_indices = np.argsort(scores)[::-1][:TOP_K]
                
                # y_pred_binary_at_k: 상위 K개를 1로 표시, 나머지를 0으로 표시
                y_pred_binary_at_k = np.zeros_like(y_true)
                if len(top_indices) > 0: # top_indices가 비어있지 않은 경우에만 할당
                    y_pred_binary_at_k[top_indices] = 1

                precision_at_k = np.sum(y_true[top_indices]) / TOP_K if TOP_K > 0 else 0.0

                current_recall = np.nan
                current_f1 = np.nan
                current_roc_auc = np.nan
                current_ap = np.nan

                if np.sum(y_true) > 0:
                    current_recall = recall_score(y_true, y_pred_binary_at_k)
                    current_f1 = f1_score(y_true, y_pred_binary_at_k)
                
                if len(np.unique(y_true)) > 1 and not np.isnan(scores).all(): # y_true에 두 클래스 이상 존재하고 scores에 NaN이 없어야 함
                    try:
                        current_roc_auc = roc_auc_score(y_true, scores)
                    except ValueError:
                        current_roc_auc = np.nan
                    try:
                        current_ap = average_precision_score(y_true, scores)
                    except ValueError:
                        current_ap = np.nan

                print(f"  {method_name} for {coin}: P@{TOP_K}={precision_at_k:.2f}, R={current_recall:.2f}, F1={current_f1:.2f}, ROC_AUC={current_roc_auc:.2f}, AP={current_ap:.2f}")

                all_ae_detection_results.append({
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
            print(f"❌ Failed to process {coin} for AE Anomaly Detection: {e}")

# 결과 저장
df_results = pd.DataFrame(all_ae_detection_results).drop_duplicates(subset=["coin", "algorithm"])
output_csv_path = os.path.join(RESULTS_DIR, "ae_anomaly_detection_results.csv")
df_results.to_csv(output_csv_path, index=False)
print(f"\n--- All AE Anomaly Detection analyses complete. Results saved to {output_csv_path} ---")