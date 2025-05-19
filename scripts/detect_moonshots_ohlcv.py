# detect_moonshots_ohlcv.py
import pandas as pd

# 문샷 감지 함수: 종가 기준 threshold 배 이상 상승한 시점 탐지
def detect_moonshots(df, threshold=2.0, window=30):
    close = df['close'].values
    moonshots = []
    for i in range(len(close) - window):
        start_price = close[i]
        future_window = close[i+1:i+1+window]
        max_future = future_window.max()
        if max_future >= start_price * threshold:
            moonshots.append((i, i + window))
    return moonshots

# 예시 사용법
if __name__ == "__main__":
    df = pd.read_csv("data/binance_ohlcv/BTCUSDT.csv")
    moonshots = detect_moonshots(df, threshold=2.0)
    print(f"Found {len(moonshots)} moonshots.")
