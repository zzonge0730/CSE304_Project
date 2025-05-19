# compress_ae_multi.py
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

FEATURE_DIR = "features/binance_top300"
LATENT_DIM = 8
EPOCHS = 50
BATCH_SIZE = 16

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
        X = np.load(x_path)

        if X.shape[0] < 10:
            continue

        print(f"🔧 Compressing {coin} with AE")
        model, encoder = build_autoencoder(X.shape[1])
        early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        model.fit(X, X, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, callbacks=[early_stop])

        Z = encoder.predict(X)
        np.save(os.path.join(FEATURE_DIR, f"Z_ae_{coin}.npy"), Z)
        print(f"✅ Saved: Z_ae_{coin}.npy")
