# fetch_binance_ohlcv.py
import requests
import pandas as pd
import time
import os
from datetime import datetime, timedelta

SAVE_DIR = "data/binance_ohlcv"
os.makedirs(SAVE_DIR, exist_ok=True)

# Binance 상위 300개 코인 심볼 받아오기 (USDT 마켓 기준)
def get_top_binance_symbols(limit=300):
    url = "https://api.binance.com/api/v3/exchangeInfo"
    res = requests.get(url)
    if res.status_code != 200:
        raise Exception("Failed to fetch Binance symbols")
    symbols = res.json()['symbols']
    usdt_pairs = [s['symbol'] for s in symbols if s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING']
    return usdt_pairs[:limit]

# OHLCV 수집 함수 (1일 간격, 최대 1000개)
def fetch_ohlcv_binance(symbol, interval='1d', start_days_ago=1000):
    url = "https://api.binance.com/api/v3/klines"
    end_time = int(time.time() * 1000)
    start_time = int((datetime.now() - timedelta(days=start_days_ago)).timestamp() * 1000)
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': start_time,
        'endTime': end_time,
        'limit': 1000
    }
    res = requests.get(url, params=params)
    if res.status_code != 200:
        print(f"❌ {symbol}: status {res.status_code}")
        return None
    raw = res.json()
    df = pd.DataFrame(raw, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'num_trades',
        'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
    ])
    df['date'] = pd.to_datetime(df['open_time'], unit='ms').dt.date
    df = df[['date', 'open', 'high', 'low', 'close', 'volume']].astype({
        'open': 'float', 'high': 'float', 'low': 'float', 'close': 'float', 'volume': 'float'
    })
    return df

if __name__ == "__main__":
    SYMBOLS = get_top_binance_symbols(300)
    for symbol in SYMBOLS:
        save_path = os.path.join(SAVE_DIR, f"{symbol}.csv")
        if os.path.exists(save_path):
            print(f"✅ Already exists: {symbol}.csv")
            continue

        print(f"📥 Fetching: {symbol}")
        df = fetch_ohlcv_binance(symbol)
        if df is not None:
            df.to_csv(save_path, index=False)
            print(f"✅ Saved: {symbol}.csv")
        time.sleep(1)
