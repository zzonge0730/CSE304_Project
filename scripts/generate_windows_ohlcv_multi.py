# generate_windows_ohlcv_multi.py
import os
import numpy as np
import pandas as pd
from detect_moonshots_ohlcv import detect_moonshots
from extend_features import extract_extended_features  # 확장 피처 정의 파일

INPUT_DIR = "../data/binance_ohlcv"
OUTPUT_DIR = "features/binance_top300"
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW_SIZE = 30
NEGATIVE_SAMPLES = 5

for filename in os.listdir(INPUT_DIR):
    if not filename.endswith(".csv"):
        continue

    coin_name = filename[:-4]
    path = os.path.join(INPUT_DIR, filename)
    df = pd.read_csv(path)

    if len(df) < WINDOW_SIZE * 3:
        print(f"{coin_name} too short, skipping")
        continue

    print(f"Processing {coin_name}...")
    features, labels = [], []

    moonshots = detect_moonshots(df, threshold=2.0, window=WINDOW_SIZE)

    # 양성 샘플 (문샷)
    for start, end in moonshots:
        close_window = df['close'].iloc[start:end]
        volume_window = df['volume'].iloc[start:end]
        if len(close_window) < WINDOW_SIZE:
            continue
        feat = extract_extended_features(close_window, volume_window)
        features.append(feat)
        labels.append(1)

    # 음성 샘플 (랜덤 시점)
    for _ in range(NEGATIVE_SAMPLES):
        i = np.random.randint(WINDOW_SIZE, len(df))
        close_window = df['close'].iloc[i - WINDOW_SIZE:i]
        volume_window = df['volume'].iloc[i - WINDOW_SIZE:i]
        feat = extract_extended_features(close_window, volume_window)
        features.append(feat)
        labels.append(0)

    if len(features) < 10:
        print(f"⚠️ {coin_name}: not enough data, skipping")
        continue

    X = np.array(features)
    y = np.array(labels)

    np.save(f"{OUTPUT_DIR}/X_{coin_name}.npy", X)
    np.save(f"{OUTPUT_DIR}/y_{coin_name}.npy", y)
    print(f"Saved {coin_name}: {X.shape}, {y.sum()} moonshots")
