import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- 설정 ---
FEATURE_DIR = "features/binance_top300"
COIN_TO_ANALYZE = "KAVAUSDT" # PCA Component를 해석할 코인 선택 (클러스터링 시각화에 사용한 코인)


FEATURE_NAMES = [
    'returns_last', 'log_returns_last',
    'ma5_last', 'ma10_last', 'ma20_last',
    'volatility_last', 'bb_width_last', 'rsi_last',
    'macd_last', 'macd_signal_last', 'adx_last', 'vol_ema_last'
]
# momentum[-30:] 과 log_volume[-30:]은 각각 30차원씩을 차지합니다.
# 이들의 이름을 'momentum_day_1', 'momentum_day_2', ... 'momentum_day_30' 등으로 명명할 수 있습니다.
# 여기서는 편의상 Generic_Feature_X 로 표시합니다.
for i in range(1, 31):
    FEATURE_NAMES.append(f'momentum_day_{i}')
for i in range(1, 31):
    FEATURE_NAMES.append(f'log_volume_day_{i}')

# 총 72차원인지 확인
if len(FEATURE_NAMES) != 72:
    print(f"경고: FEATURE_NAMES 리스트의 길이가 72가 아닙니다. 현재 {len(FEATURE_NAMES)} 차원.")
    print("extend_features.py의 extract_extended_features 함수 반환 순서와 정확히 일치시켜주세요.")
    exit()

# --- 데이터 로드 및 준비 ---
x_raw_path = os.path.join(FEATURE_DIR, f"X_{COIN_TO_ANALYZE}.npy")

try:
    X_raw = np.load(x_raw_path)
except FileNotFoundError:
    print(f"오류: {COIN_TO_ANALYZE}에 대한 원본 피처 파일을 찾을 수 없습니다.")
    exit()

if len(X_raw) < 10:
    print(f"⚠️ {COIN_TO_ANALYZE}: 샘플 수가 부족합니다 ({len(X_raw)} 샘플). PCA 해석에 충분하지 않을 수 있습니다.")
    exit()

scaler = StandardScaler()
X_scaled_raw = scaler.fit_transform(X_raw)

# PCA 수행 (2차원)
pca_model = PCA(n_components=2, random_state=42)
pca_model.fit(X_scaled_raw)

# --- PCA Component Loadings 해석 ---
print(f"\n--- {COIN_TO_ANALYZE} PCA Component Loadings ---")
print(f"Explained Variance Ratio: {pca_model.explained_variance_ratio_}")
print(f"Total Explained Variance (PC1+PC2): {pca_model.explained_variance_ratio_.sum():.3f}")

for i, component_loadings in enumerate(pca_model.components_):
    print(f"\nPCA Component {i+1} (Explains {pca_model.explained_variance_ratio_[i]:.3f} variance):")
    # 가중치(절댓값)가 큰 순서대로 정렬하여 출력
    sorted_loadings = sorted(zip(FEATURE_NAMES, component_loadings), key=lambda x: abs(x[1]), reverse=True)
    
    # 상위 N개 피처만 출력 (여기서는 상위 5개)
    for feature_name, loading in sorted_loadings[:5]:
        print(f"  {feature_name}: {loading:.4f}")

print("\n--- PCA Component Loadings 해석 완료 ---")