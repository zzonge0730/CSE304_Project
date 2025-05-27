# compress_ae_multi.py
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings

warnings.filterwarnings("ignore") # 텐서플로우 경고 무시

FEATURE_DIR = "features/binance_top300"
LATENT_DIM = 8
EPOCHS = 50
BATCH_SIZE = 16
# 최소 샘플 수 조건 완화: 5개 샘플도 학습 시도
MIN_SAMPLES_FOR_AE_TRAINING = 5 # 10 -> 5로 변경 (더 많은 코인 포함)

# Autoencoder 구성 함수
def build_autoencoder(input_dim):
    inp = Input(shape=(input_dim,))
    encoded = Dense(32, activation='relu')(inp)
    encoded = Dense(LATENT_DIM, activation='relu')(encoded)
    decoded = Dense(32, activation='relu')(encoded)
    out = Dense(input_dim, activation='linear')(decoded)
    model = Model(inputs=inp, outputs=out)
    encoder = Model(inputs=inp, outputs=encoded)
    model.compile(optimizer=Adam(0.001), loss='mse')
    return model, encoder

for file in os.listdir(FEATURE_DIR):
    if file.startswith("X_") and file.endswith(".npy"):
        coin = file[2:-4]
        x_path = os.path.join(FEATURE_DIR, f"X_{coin}.npy")
        
        # 이미 압축된 파일이 존재하면 건너뛰기 (시간 절약)
        z_ae_path = os.path.join(FEATURE_DIR, f"Z_ae_{coin}.npy")
        if os.path.exists(z_ae_path):
            print(f"Already compressed: Z_ae_{coin}.npy")
            continue

        try:
            X = np.load(x_path)

            if X.shape[0] < MIN_SAMPLES_FOR_AE_TRAINING: # 완화된 조건 적용
                print(f"⚠️ {coin} skipped AE compression: not enough samples ({X.shape[0]} < {MIN_SAMPLES_FOR_AE_TRAINING})")
                continue

            print(f"🔧 Compressing {coin} with AE (samples: {X.shape[0]}, input_dim: {X.shape[1]})")
            model, encoder = build_autoencoder(X.shape[1])
            early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
            
            # AE 학습 중 오류 발생 가능성이 있으므로 try-except 추가
            model.fit(X, X, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, callbacks=[early_stop])

            Z = encoder.predict(X, verbose=0)
            np.save(z_ae_path, Z)
            print(f"Saved: {z_ae_path}")
        except Exception as e:
            print(f"Failed to compress {coin} with AE: {e}")