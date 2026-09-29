# Cryptocurrency Anomaly-Period Detection (CSE304 Data Mining)

Finds unusual price and volume periods in Binance USDT-pair markets and compares dimension-reduction and anomaly-detection methods under one protocol.

## What it does
- Data: Binance OHLCV for 242 assets, 72 technical indicators (data/binance_ohlcv, scripts/extend_features.py)
- Dimension reduction: PCA, sparse PCA, autoencoder (scripts/compress_*.py)
- Detection: LOF, Isolation Forest, autoencoder reconstruction error
- Best result: autoencoder reconstruction error, ROC-AUC 0.786, Precision@10 0.932

## Limits
This detects anomalous periods only. It does not explain their causes and does not forecast prices.

## Layout
- data/: price CSVs
- scripts/: feature extension, compression and detection scripts
