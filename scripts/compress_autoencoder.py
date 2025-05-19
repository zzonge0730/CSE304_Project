import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Load data
X = np.load("features/X.npy")
X_tensor = torch.tensor(X, dtype=torch.float32)

# Define Autoencoder
class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(60, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # bottleneck
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 60)
        )
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

model = AE()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train
loader = DataLoader(TensorDataset(X_tensor, X_tensor), batch_size=16, shuffle=True)
for epoch in range(30):
    for batch_x, _ in loader:
        x_hat = model(batch_x)
        loss = criterion(x_hat, batch_x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Extract 2D latent representation
with torch.no_grad():
    Z = model.encoder(X_tensor).numpy()
np.save("features/Z_ae.npy", Z)
print("✅ Autoencoder compression done:", Z.shape)
