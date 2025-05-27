# compress_sparse_pca_multi.py
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import SparsePCA
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

FEATURE_DIR = "features/binance_top300"
# 결과 저장 디렉토리
OUTPUT_DIR = FEATURE_DIR 

# Sparse PCA 설정
# n_components: 압축할 차원 수 (예: 8, 16 등, input_dim보다 작게)
# alpha: 희소성 강도 (클수록 더 희소해짐. 0.1, 0.5, 1.0 등 테스트)
# random_state: 재현성을 위한 시드
SPARSE_PCA_N_COMPONENTS = 8 # AutoEncoder의 LATENT_DIM과 동일하게 설정하여 비교하기 용이
SPARSE_PCA_ALPHA = 0.5 # 튜닝이 필요한 중요한 파라미터입니다.
SPARSE_PCA_RANDOM_STATE = 42

# 최소 샘플 수 (너무 적으면 Sparse PCA 학습 불가)
MIN_SAMPLES_FOR_SPARSE_PCA = 10 # 기본 10개로 유지 (AE 압축과 동일하게 맞춤)

for file in os.listdir(FEATURE_DIR):
    if file.startswith("X_") and file.endswith(".npy"):
        coin = file[2:-4]
        x_path = os.path.join(FEATURE_DIR, f"X_{coin}.npy")

        # 이미 압축된 파일이 존재하면 건너뛰기
        z_sparse_pca_path = os.path.join(OUTPUT_DIR, f"Z_sparse_pca_{coin}.npy")
        if os.path.exists(z_sparse_pca_path):
            print(f"✅ Already compressed: {z_sparse_pca_path}")
            continue

        try:
            X_raw = np.load(x_path)

            if X_raw.shape[0] < MIN_SAMPLES_FOR_SPARSE_PCA:
                print(f"⚠️ {coin} skipped Sparse PCA compression: not enough samples ({X_raw.shape[0]} < {MIN_SAMPLES_FOR_SPARSE_PCA})")
                continue

            print(f"🔧 Compressing {coin} with Sparse PCA (samples: {X_raw.shape[0]}, input_dim: {X_raw.shape[1]})")

            # 데이터 스케일링 (Sparse PCA도 스케일링된 데이터에 적용하는 것이 좋음)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_raw)

            # Sparse PCA 모델 학습 및 압축
            sparse_pca = SparsePCA(
                n_components=SPARSE_PCA_N_COMPONENTS,
                alpha=SPARSE_PCA_ALPHA,
                random_state=SPARSE_PCA_RANDOM_STATE
            )
            Z_sparse_pca = sparse_pca.fit_transform(X_scaled) # 압축된 피처

            np.save(z_sparse_pca_path, Z_sparse_pca)
            print(f"Saved: {z_sparse_pca_path}")

        except Exception as e:
            print(f"Failed to compress {coin} with Sparse PCA: {e}")