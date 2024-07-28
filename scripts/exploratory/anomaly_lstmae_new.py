import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

# Load a synthetic dataset
data_url = "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv"
data = pd.read_csv(data_url)
data['value'] = MinMaxScaler().fit_transform(data['value'].values.reshape(-1, 1))
data = data['value'].values

# Visualize the data
plt.plot(data)
# plt.show(block=True)  # Make plt.show() blocking

# Create sequences for LSTM input
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
    return np.array(sequences)

SEQ_LENGTH = 50
sequences = create_sequences(data, SEQ_LENGTH)
train_size = int(0.8 * len(sequences))
train_sequences, test_sequences = sequences[:train_size], sequences[train_size:]

train_sequences = np.expand_dims(train_sequences, -1)  # Add feature dimension
test_sequences = np.expand_dims(test_sequences, -1)    # Add feature dimension

# Remove the extra dimension added
train_sequences = train_sequences.reshape((train_sequences.shape[0], SEQ_LENGTH, -1))
test_sequences = test_sequences.reshape((test_sequences.shape[0], SEQ_LENGTH, -1))

# Convert data to torch.float32
train_sequences = torch.tensor(train_sequences, dtype=torch.float32)
test_sequences = torch.tensor(test_sequences, dtype=torch.float32)

# Define the LSTM autoencoder model in PyTorch
class LSTM_Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTM_Autoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        x, (hidden, cell) = self.encoder(x)
        x = hidden.repeat(x.size(1), 1, 1).permute(1, 0, 2)  # Repeat hidden state across sequence length
        x, _ = self.decoder(x)
        return x

input_dim = train_sequences.shape[2]
hidden_dim = 64

model = LSTM_Autoencoder(input_dim, hidden_dim).cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define num_epochs
num_epochs = 100

# Create DataLoader
train_loader = torch.utils.data.DataLoader(train_sequences, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(test_sequences, batch_size=16, shuffle=False)

# Train the model
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for batch in train_loader:
        batch = batch.cuda()
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.cuda()
            output = model(batch)
            loss = criterion(output, batch)
            total_val_loss += loss.item()
    
    avg_train_loss = total_train_loss / len(train_loader)
    avg_val_loss = total_val_loss / len(val_loader)
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Detect anomalies based on reconstruction error
def detect_anomalies(model, sequences, threshold):
    model.eval()
    with torch.no_grad():
        reconstructions = model(sequences.cuda())
        mse = torch.mean((sequences.cuda() - reconstructions) ** 2, dim=(1, 2)).cpu().numpy()
    return mse > threshold, mse

with torch.no_grad():
    reconstructions = model(test_sequences.cuda())
    mse = torch.mean((test_sequences.cuda() - reconstructions) ** 2, dim=(1, 2)).cpu().numpy()

# Set a threshold for anomaly detection
threshold = np.percentile(mse, 95)
anomalies, reconstruction_errors = detect_anomalies(model, test_sequences, threshold)

# Visualize reconstruction errors and detected anomalies
plt.figure(figsize=(10, 6))
plt.plot(reconstruction_errors, label='Reconstruction Error')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.legend()
plt.show(block=True)  # Make plt.show() blocking

# # Since 'anomaly' column is missing, we skip the evaluation part based on ground truth.
# # Evaluate performance if ground truth is available
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ground_truth = pd.read_csv(data_url)['anomaly'].values[SEQ_LENGTH:][train_size:]
# print(f'Accuracy: {accuracy_score(ground_truth, anomalies):.4f}')
# print(f'Precision: {precision_score(ground_truth, anomalies):.4f}')
# print(f'Recall: {recall_score(ground_truth, anomalies):.4f}')
# print(f'F1 Score: {f1_score(ground_truth, anomalies):.4f}')
