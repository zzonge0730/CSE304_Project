# top_50_fetch_prices.py
import requests
import pandas as pd
import time
import os

DATA_DIR = "data/top50"
os.makedirs(DATA_DIR, exist_ok=True)

# 1. 상위 N개 코인 ID 가져오기
def get_top_coin_ids(n=50):
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        'vs_currency': 'usd',
        'order': 'market_cap_desc',
        'per_page': n,
        'page': 1,
        'sparkline': False
    }
    res = requests.get(url, params=params)
    if res.status_code != 200:
        print("❌ Failed to fetch coin list:", res.status_code)
        return []
    coins = res.json()
    top_ids = [coin['id'] for coin in coins]
    print(f"✅ Fetched top {n} coin IDs.")
    return top_ids

# 2. 코인별 가격 데이터 수집
def fetch_coin_history(coin_id, days=365):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {'vs_currency': 'usd', 'days': days, 'interval': 'daily'}
    try:
        res = requests.get(url, params=params)
        if res.status_code != 200:
            print(f"❌ {coin_id}: status {res.status_code}")
            return None
        data = res.json()
        df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
        return df[['date', 'price']]
    except Exception as e:
        print(f"⚠️ {coin_id}: error {e}")
        return None

# 3. 전체 실행
if __name__ == "__main__":
    coin_ids = get_top_coin_ids(50)

    for coin in coin_ids:
        print(f"📥 Fetching: {coin}")
        df = fetch_coin_history(coin)
        if df is not None:
            df.to_csv(f"{DATA_DIR}/{coin}.csv", index=False)
            print(f"✅ Saved: {coin}.csv")
        time.sleep(60)  # Rate limit 회피
