import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings

# 경고 메시지 무시 (예: KMeans 수렴 경고 등)
warnings.filterwarnings("ignore")

# --- 설정 ---
FEATURE_DIR = "features/binance_top300"
RESULTS_DIR = "clustering_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# K-Means 설정
KMEANS_N_CLUSTERS_RANGE = range(2, 6) # 2개부터 5개까지 클러스터 개수 테스트
KMEANS_RANDOM_STATE = 42
KMEANS_N_INIT = 10 # n_init 추가

# DBSCAN 설정 (데이터에 따라 튜닝 필요)
DBSCAN_EPS = 0.5
DBSCAN_MIN_SAMPLES = 5

# 최소 샘플 수 (데이터가 너무 적으면 클러스터링이 불안정)
MIN_SAMPLES_FOR_CLUSTERING = 10 # generate_windows_ohlcv_multi.py와 동일하게 10으로 설정

# --- 메인 실행 ---
all_results_list = []

# FEATURE_DIR 내의 모든 X_*.npy 파일 탐색
for filename in os.listdir(FEATURE_DIR):
    if filename.startswith("X_") and filename.endswith(".npy"):
        coin_name = filename[2:-4] # X_ 제거, .npy 제거

        x_raw_path = os.path.join(FEATURE_DIR, filename)
        z_ae_path = os.path.join(FEATURE_DIR, f"Z_ae_{coin_name}.npy")
        y_label_path = os.path.join(FEATURE_DIR, f"y_{coin_name}.npy")

        # 필요한 모든 파일이 존재하는지 확인
        if not os.path.exists(x_raw_path) or \
           not os.path.exists(z_ae_path) or \
           not os.path.exists(y_label_path):
            print(f"⚠️ Skipping {coin_name}: Missing one or more feature/label files.")
            continue

        try:
            X_raw = np.load(x_raw_path)
            X_compressed_ae = np.load(z_ae_path)
            y_labels = np.load(y_label_path)

            if len(X_raw) < MIN_SAMPLES_FOR_CLUSTERING:
                print(f"⚠️ Skipping {coin_name}: Not enough samples ({len(X_raw)}).")
                continue

            print(f"\n--- Processing {coin_name} ---")

            # 압축 전/후 데이터를 담을 딕셔너리
            data_sets = {
                "raw": {"data": X_raw, "name": "Raw Features"},
                "compressed_ae": {"data": X_compressed_ae, "name": "AE Compressed Features"}
            }

            for data_type, data_info in data_sets.items():
                data = data_info["data"]
                data_name = data_info["name"]

                # 데이터 스케일링
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(data)

                # --- K-Means 클러스터링 ---
                best_kmeans_score = -2 # 실루엣 점수 범위(-1 ~ 1)를 벗어나는 값으로 초기화
                best_kmeans_k = np.nan
                best_kmeans_labels = None

                for k in KMEANS_N_CLUSTERS_RANGE:
                    if k >= len(X_scaled):
                        continue
                    try:
                        kmeans = KMeans(n_clusters=k, random_state=KMEANS_RANDOM_STATE, n_init=KMEANS_N_INIT)
                        kmeans_labels = kmeans.fit_predict(X_scaled)
                        if len(np.unique(kmeans_labels)) > 1: # 클러스터가 1개 초과인 경우만 평가
                            silhouette_val = silhouette_score(X_scaled, kmeans_labels)
                            if silhouette_val > best_kmeans_score:
                                best_kmeans_score = silhouette_val
                                best_kmeans_k = k
                                best_kmeans_labels = kmeans_labels
                    except Exception: # K-Means 오류 발생 시 (예: k가 너무 커서)
                        continue # 이 k 값은 건너뛰고 다음 k로
                
                # K-Means 결과 저장
                row_data = {
                    "coin": coin_name,
                    "feature_type": data_type,
                    "num_samples": len(X_scaled),
                    "num_moonshots": int(np.sum(y_labels)),
                    "algorithm": "KMeans",
                    "best_k": best_kmeans_k,
                    "silhouette_score": round(best_kmeans_score, 3) if best_kmeans_score != -2 else np.nan # -2는 유효한 클러스터가 없었음 의미
                }
                all_results_list.append(row_data)
                print(f"  {data_name} - KMeans (best k={best_kmeans_k}): Silhouette = {row_data['silhouette_score']:.3f}")

                # --- DBSCAN 클러스터링 ---
                try:
                    dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
                    dbscan_labels = dbscan.fit_predict(X_scaled)
                    
                    silhouette_dbscan = np.nan
                    # 클러스터가 1개 이상이고 노이즈가 아닌 유효한 클러스터가 있을 경우에만 실루엣 계산
                    if len(np.unique(dbscan_labels)) > 1 and -1 in np.unique(dbscan_labels) and len(np.unique(dbscan_labels)) > 2: # 노이즈(-1) 제외한 클러스터가 2개 이상
                         silhouette_dbscan = silhouette_score(X_scaled[dbscan_labels != -1], dbscan_labels[dbscan_labels != -1])
                    elif len(np.unique(dbscan_labels)) > 1 and -1 not in np.unique(dbscan_labels): # 노이즈 없이 2개 이상 클러스터
                        silhouette_dbscan = silhouette_score(X_scaled, dbscan_labels)
                    
                    row_data = {
                        "coin": coin_name,
                        "feature_type": data_type,
                        "num_samples": len(X_scaled),
                        "num_moonshots": int(np.sum(y_labels)),
                        "algorithm": "DBSCAN",
                        "best_k": np.nan, # DBSCAN은 k 개념이 없음
                        "silhouette_score": round(silhouette_dbscan, 3) if not np.isnan(silhouette_dbscan) else np.nan
                    }
                    all_results_list.append(row_data)
                    print(f"  {data_name} - DBSCAN (eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES}): Silhouette = {row_data['silhouette_score']:.3f}")

                except Exception as e:
                    print(f"❌ {data_name} - DBSCAN failed: {e}")
                    row_data = {
                        "coin": coin_name,
                        "feature_type": data_type,
                        "num_samples": len(X_scaled),
                        "num_moonshots": int(np.sum(y_labels)),
                        "algorithm": "DBSCAN",
                        "best_k": np.nan,
                        "silhouette_score": np.nan
                    }
                    all_results_list.append(row_data)

        except Exception as e:
            print(f"❌ Failed to process {coin_name}: {e}")

# 최종 결과 저장
df_results = pd.DataFrame(all_results_list)
output_csv_path = os.path.join(RESULTS_DIR, "clustering_comparison_results.csv")
df_results.to_csv(output_csv_path, index=False)
print(f"\n--- All clustering analyses complete. Results saved to {output_csv_path} ---")