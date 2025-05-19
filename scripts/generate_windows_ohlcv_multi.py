# generate_windows_ohlcv_multi.py
import os
import numpy as np
import pandas as pd
from detect_moonshots_ohlcv import detect_moonshots

INPUT_DIR = "../data/binance_ohlcv"
OUTPUT_DIR = "features/binance_top300"
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW_SIZE = 30
NEGATIVE_SAMPLES = 5

# 피처 벡터 추출: 수익률 + 변동성
def extract_features(df_window):
    close = df_window['close'].pct_change().fillna(0).values
    volatility = pd.Series(close).rolling(window=7).std().fillna(0).values
    return np.concatenate([close[-30:], volatility[-30:]])

for filename in os.listdir(INPUT_DIR):
    if not filename.endswith(".csv"):
        continue

    coin_name = filename[:-4]
    path = os.path.join(INPUT_DIR, filename)
    df = pd.read_csv(path)

    if len(df) < WINDOW_SIZE * 3:
        print(f"❌ {coin_name} too short, skipping")
        continue

    print(f"🔄 Processing {coin_name}...")
    features, labels = [], []

    moonshots = detect_moonshots(df, threshold=2.0, window=WINDOW_SIZE)

    for start, end in moonshots:
        window = df.iloc[start:end]
        feat = extract_features(window)
        features.append(feat)
        labels.append(1)

    for _ in range(NEGATIVE_SAMPLES):
        i = np.random.randint(WINDOW_SIZE, len(df) - WINDOW_SIZE)
        window = df.iloc[i - WINDOW_SIZE:i]
        feat = extract_features(window)
        features.append(feat)
        labels.append(0)

    if len(features) < 10:
        print(f"⚠️ {coin_name}: not enough data, skipping")
        continue

    X = np.array(features)
    y = np.array(labels)

    np.save(f"{OUTPUT_DIR}/X_{coin_name}.npy", X)
    np.save(f"{OUTPUT_DIR}/y_{coin_name}.npy", y)
    print(f"✅ Saved {coin_name}: {X.shape}, {y.sum()} moonshots")
