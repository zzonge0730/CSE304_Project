import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# --- 설정 ---
FEATURE_DIR = "features/binance_top300"
# 시각화할 코인 심볼 목록 (위에서 추천된 코인들)
COINS_TO_VISUALIZE_LIST = [
    "FIDAUSDT",
    "DODOUSDT",
    "WLDUSDT",
    "XLMUSDT",
    "KAVAUSDT"
]

# DBSCAN 설정 (데이터 및 Feature Type에 따라 튜닝 필요)
# 이 값들은 clustering_comparison_results.csv에서 최적화되지 않았습니다.
# 각 코인별로 시각화하면서 최적의 DBSCAN 클러스터를 확인하고 싶다면
# 이 값을 수동으로 조정하거나, 별도의 튜닝 루프를 만들 수 있습니다.
DBSCAN_EPS_RAW = 0.1
DBSCAN_MIN_SAMPLES_RAW = 3
DBSCAN_EPS_COMPRESSED = 0.5
DBSCAN_MIN_SAMPLES_COMPRESSED = 5

# 시각화용 PCA 차원
PCA_COMPONENTS_FOR_VIZ = 2

# 이미지 저장 경로
IMAGE_SAVE_DIR = "visualization_output"
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

# --- 결과 파일 로드 (K-Means k 값 자동 설정을 위함) ---
try:
    results_df = pd.read_csv('clustering_results/clustering_comparison_results.csv')
    print("✅ clustering_comparison_results.csv 로드 완료.")
except FileNotFoundError:
    print("❌ 오류: clustering_results/clustering_comparison_results.csv 파일을 찾을 수 없습니다.")
    print("먼저 모든 코인에 대한 클러스터링 결과를 생성해야 합니다.")
    exit()

# --- 데이터 로드 및 준비 함수 ---
def load_and_prepare_data(coin_name, feature_type):
    if feature_type == 'raw':
        data_path = os.path.join(FEATURE_DIR, f"X_{coin_name}.npy")
    elif feature_type == 'compressed_ae':
        data_path = os.path.join(FEATURE_DIR, f"Z_ae_{coin_name}.npy")
    else:
        raise ValueError("feature_type must be 'raw' or 'compressed_ae'")

    y_path = os.path.join(FEATURE_DIR, f"y_{coin_name}.npy")

    if not os.path.exists(data_path) or not os.path.exists(y_path):
        print(f"❌ 오류: {coin_name} ({feature_type})에 대한 데이터 파일을 찾을 수 없습니다.")
        return None, None, None

    data = np.load(data_path)
    labels = np.load(y_path)

    if len(data) < 5: # 최소 샘플 수 확인
        print(f"⚠️ {coin_name} ({feature_type}): 샘플 수가 부족합니다 ({len(data)} 샘플).")
        return None, None, None

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    pca_viz = PCA(n_components=PCA_COMPONENTS_FOR_VIZ)
    data_2d_viz = pca_viz.fit_transform(scaled_data)

    return scaled_data, data_2d_viz, labels

# --- 클러스터링 및 시각화 함수 ---
def perform_and_visualize_clustering(coin_name, X_scaled, X_2d_viz, y_moonshot, title_prefix,
                                     kmeans_n_clusters, dbscan_eps, dbscan_min_samples):
    # K-Means 클러스터링
    kmeans_labels = None
    kmeans_score = np.nan
    if kmeans_n_clusters >= 2 and kmeans_n_clusters < len(X_scaled):
        try:
            kmeans = KMeans(n_clusters=kmeans_n_clusters, random_state=42, n_init=10)
            kmeans_labels = kmeans.fit_predict(X_scaled)
            if len(np.unique(kmeans_labels)) > 1:
                kmeans_score = silhouette_score(X_scaled, kmeans_labels)
                print(f"{title_prefix} - KMeans Silhouette Score (k={kmeans_n_clusters}): {kmeans_score:.3f}")
            else:
                print(f"{title_prefix} - KMeans (k={kmeans_n_clusters}): 단일 클러스터만 형성되어 실루엣 점수를 계산할 수 없습니다.")
        except Exception as e:
            print(f"❌ {title_prefix} - KMeans 실패: {e}")
    else:
        print(f"⚠️ {title_prefix} - KMeans (k={kmeans_n_clusters}): 클러스터 개수가 유효하지 않거나 샘플 수보다 많습니다.")

    # DBSCAN 클러스터링
    dbscan_labels = None
    dbscan_score = np.nan
    if dbscan_eps is not None and dbscan_min_samples is not None:
        try:
            dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
            dbscan_labels = dbscan.fit_predict(X_scaled)

            n_clusters_dbscan = len(np.unique(dbscan_labels))

            if n_clusters_dbscan > 1 and -1 not in np.unique(dbscan_labels):
                dbscan_score = silhouette_score(X_scaled, dbscan_labels)
                print(f"{title_prefix} - DBSCAN Silhouette Score (eps={dbscan_eps}, min_samples={dbscan_min_samples}): {dbscan_score:.3f}")
            elif n_clusters_dbscan > 1 and -1 in np.unique(dbscan_labels) and len(np.unique(dbscan_labels[dbscan_labels != -1])) > 1:
                dbscan_score = silhouette_score(X_scaled[dbscan_labels != -1], dbscan_labels[dbscan_labels != -1])
                print(f"{title_prefix} - DBSCAN Silhouette Score (노이즈 제외, eps={dbscan_eps}, min_samples={dbscan_min_samples}): {dbscan_score:.3f}")
            else:
                print(f"{title_prefix} - DBSCAN: 단일 클러스터만 형성되었거나 모든 점이 노이즈로 분류되어 실루엣 점수를 계산할 수 없습니다. (클러스터 수: {n_clusters_dbscan})")

        except Exception as e:
            print(f"❌ {title_prefix} - DBSCAN 실패: {e}")

    # 시각화 플롯 생성
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f'{coin_name} 클러스터링 비교: {title_prefix}', fontsize=16)

    if kmeans_labels is not None:
        sns.scatterplot(x=X_2d_viz[:, 0], y=X_2d_viz[:, 1], hue=kmeans_labels, style=y_moonshot,
                        palette='viridis', legend='full', s=70, alpha=0.8, ax=axes[0])
        axes[0].set_title(f'K-Means (k={kmeans_n_clusters}, Silhouette: {kmeans_score:.3f})')
        axes[0].set_xlabel(f'PCA Component 1')
        axes[0].set_ylabel(f'PCA Component 2')
        axes[0].grid(True, linestyle='--', alpha=0.6)
    else:
        axes[0].set_title(f'K-Means - 유효한 클러스터 없음')
        axes[0].text(0.5, 0.5, '데이터를 표시할 수 없습니다', horizontalalignment='center', verticalalignment='center', transform=axes[0].transAxes)

    if dbscan_labels is not None:
        sns.scatterplot(x=X_2d_viz[:, 0], y=X_2d_viz[:, 1], hue=dbscan_labels, style=y_moonshot,
                        palette='viridis', legend='full', s=70, alpha=0.8, ax=axes[1])
        axes[1].set_title(f'DBSCAN (eps={dbscan_eps}, min_samples={dbscan_min_samples}, Silhouette: {dbscan_score:.3f})')
        axes[1].set_xlabel(f'PCA Component 1')
        axes[1].set_ylabel(f'PCA Component 2')
        axes[1].grid(True, linestyle='--', alpha=0.6)
    else:
        axes[1].set_title(f'DBSCAN - 유효한 클러스터 없음')
        axes[1].text(0.5, 0.5, '데이터를 표시할 수 없습니다', horizontalalignment='center', verticalalignment='center', transform=axes[1].transAxes)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 이미지 파일로 저장
    figure_filename = f"{coin_name}_{title_prefix.replace(' ', '_').replace('(', '').replace(')', '')}_Clustering.png"
    plt.savefig(os.path.join(IMAGE_SAVE_DIR, figure_filename))
    print(f"🖼️ 이미지 저장 완료: {os.path.join(IMAGE_SAVE_DIR, figure_filename)}")
    plt.close(fig) # 창을 닫아 메모리 확보

# --- 메인 실행 루프 ---
for coin_symbol in COINS_TO_VISUALIZE_LIST:
    print(f"\n--- {coin_symbol} 코인 시각화 시작 ---")

    # K-Means k 값 자동 로드
    coin_results = results_df[results_df['coin'] == coin_symbol]
    kmeans_k_raw = coin_results[(coin_results['algorithm'] == 'KMeans') & (coin_results['feature_type'] == 'raw')]['best_k'].iloc[0] if not coin_results[(coin_results['algorithm'] == 'KMeans') & (coin_results['feature_type'] == 'raw')]['best_k'].empty else 3
    kmeans_k_compressed = coin_results[(coin_results['algorithm'] == 'KMeans') & (coin_results['feature_type'] == 'compressed_ae')]['best_k'].iloc[0] if not coin_results[(coin_results['algorithm'] == 'KMeans') & (coin_results['feature_type'] == 'compressed_ae')]['best_k'].empty else 3

    # 1. 원본 피처 데이터 로드 및 클러스터링 시각화
    X_raw_scaled, X_raw_2d_viz, y_moonshot_raw = load_and_prepare_data(coin_symbol, 'raw')
    if X_raw_scaled is not None:
        perform_and_visualize_clustering(coin_symbol, X_raw_scaled, X_raw_2d_viz, y_moonshot_raw,
                                         "원본 피처 (Raw Features)", int(kmeans_k_raw),
                                         DBSCAN_EPS_RAW, DBSCAN_MIN_SAMPLES_RAW)

    # 2. 압축된 피처 데이터 로드 및 클러스터링 시각화
    X_compressed_scaled, X_compressed_2d_viz, y_moonshot_compressed = load_and_prepare_data(coin_symbol, 'compressed_ae')
    if X_compressed_scaled is not None:
        perform_and_visualize_clustering(coin_symbol, X_compressed_scaled, X_compressed_2d_viz, y_moonshot_compressed,
                                         "AE 압축 피처 (AE Compressed Features)", int(kmeans_k_compressed),
                                         DBSCAN_EPS_COMPRESSED, DBSCAN_MIN_SAMPLES_COMPRESSED)

print(f"\n--- 모든 코인 시각화 완료. 이미지 파일은 '{IMAGE_SAVE_DIR}' 디렉토리에 저장되었습니다 ---")