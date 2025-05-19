import numpy as np
from sklearn.decomposition import PCA
import os

# Load features
X = np.load("features/X.npy")
y = np.load("features/y.npy")

# Apply PCA
pca = PCA(n_components=2)  # or 5
Z = pca.fit_transform(X)

# 저장
np.save("features/Z_pca.npy", Z)

print("✅ PCA complete. Shape:", Z.shape)
