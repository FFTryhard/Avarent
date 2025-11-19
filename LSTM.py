import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import csv

class LSTMAutoencoder(nn.Module):
    def __init__(self, feature_size, hidden_size, latent_size, num_layers=1):
        super().__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers

        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Project final hidden state to latent space
        self.hidden2latent = nn.Linear(hidden_size, latent_size)

        # Decoder: from latent -> hidden_size -> sequence
        self.latent2hidden = nn.Linear(latent_size, hidden_size)
        self.decoder_lstm = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.output_layer = nn.Linear(hidden_size, feature_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Encode
        _, (h_n, _) = self.encoder_lstm(x)  # h_n: [num_layers, batch, hidden_size]
        h_last = h_n[-1]                     # last layer hidden state [batch, hidden_size]
        latent = self.hidden2latent(h_last)  # [batch, latent_size]

        # Decode
        hidden_dec = self.latent2hidden(latent).unsqueeze(0)  # [1, batch, hidden_size]
        c0 = torch.zeros_like(hidden_dec)                      # initial cell state

        # Feed zeros as input to decoder (could be learned start token)
        dec_input = torch.zeros(batch_size, seq_len, self.feature_size, device=x.device)
        dec_output, _ = self.decoder_lstm(dec_input, (hidden_dec, c0))
        out = self.output_layer(dec_output)  # [batch, seq_len, feature_size]
        return out, latent

# Load CSV
df = pd.read_csv("C:/Users/H/Desktop/C++ Tryharding/.csv files/data.csv", header=0)

# Optional: display tweaks
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.precision', 6)

#print(df)

# --- NEW: sliding window sequences ---
X = df.iloc[:, 1:].values  # drop timestep column -> [timesteps, features]

window_size = 30
stride = 3
sequences = []

for start in range(0, X.shape[0] - window_size + 1, stride):
    sequences.append(X[start:start + window_size])

# Convert list -> NumPy -> Torch tensor
sequences = np.array(sequences, dtype=np.float32)   # [num_sequences, window_size, features]
sequences = torch.from_numpy(sequences)             # torch tensor
#print("Sequences shape:", sequences.shape)

# --- ACCESS THE BLOCKS DIRECTLY ---
#print("\nFirst sequence (rows 0–29):")
#print(sequences[0])  # shape [30,7]

#print("\nFirst timestep of first sequence:")
#print(sequences[0][0])  # shape [7]

#print("\nSingle value [seq0, time0, feature3]:")
#print(sequences[0][0][3].item())  # scalar float

# --- batching for LSTM ---
batch_size = 8 # Bigger batch stabilizes gradients. Tiny batches on tiny data = noisy training.
dataset = TensorDataset(sequences)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)



# Hyperparameters
feature_size = 7       # your 7 streams
hidden_size = 16       # Adjust. Hidden size controls how much “memory” the LSTM has per timestep. Too high → overfits tiny dataset. Too low → can’t model anything. 16–24 is enough to capture patterns but prevents memorization.
latent_size = 4        # Compressed representation. This is the bottleneck. The smaller it is, the more compression, the more the autoencoder is forced to learn essential structure, not just memorize. Right now 8 is fine but a little smaller will make it generalize better.
num_layers = 1
lr = 1e-3
epochs = 50

model = LSTMAutoencoder(feature_size, hidden_size, latent_size, num_layers)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# Open CSV file in write mode
csv_file_path = "C:/Users/H\Desktop/C++ Tryharding/.csv files/RE/epochs.csv"
with open(csv_file_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["loss"])  # header

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            x = batch[0]  # [batch, seq_len, features]
            optimizer.zero_grad()
            recon, latent = model(x)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)  # sum over batch

        # Average loss for the epoch
        avg_loss = epoch_loss / len(dataset)

        # Write epoch and loss to CSV
        writer.writerow([avg_loss])

        # Optional: still print if you want quick feedback
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")