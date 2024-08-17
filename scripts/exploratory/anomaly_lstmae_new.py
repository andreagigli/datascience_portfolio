import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


# Define hyperparameters
LOSSES = {
    "mse": nn.MSELoss(), 
}

HPARAMS = {
    'learning_rate': 0.001,
    'hidden_dim': 64,
    'batch_size': 32,
    'num_epochs': 100,
    "loss": "mse",
}


class TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length, padding_value=0):
        self.data = data
        self.sequence_length = sequence_length
        self.padding_value = padding_value

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if index >= self.sequence_length:
            return self.data[index - self.sequence_length:index]
        else:
            # Pad the sequence if index is less than sequence_length
            pad_size = self.sequence_length - index
            padded_sequence = torch.full((pad_size,), self.padding_value, dtype=torch.float32)
            return torch.cat((padded_sequence, self.data[:index]), dim=0)


# Load a synthetic dataset
data_url = "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv"
data = pd.read_csv(data_url)
data['value'] = MinMaxScaler().fit_transform(data['value'].values.reshape(-1, 1))
data_tensor = torch.tensor(data['value'], dtype=torch.float32).unsqueeze(-1)  # Add feature dimension

# Define the TimeSeriesDataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length, padding_value=0):
        self.data = data
        self.sequence_length = sequence_length
        self.padding_value = padding_value

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if index >= self.sequence_length:
            return self.data[index - self.sequence_length:index]
        else:
            pad_size = self.sequence_length - index
            padded_sequence = torch.full((pad_size, 1), self.padding_value, dtype=torch.float32)
            return torch.cat((padded_sequence, self.data[:index]), dim=0)

SEQ_LENGTH = 50
dataset = TimeSeriesDataset(data_tensor, SEQ_LENGTH)

# Split data into training and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=HPARAMS["batch_size"], shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=HPARAMS["batch_size"], shuffle=False)


# Define the LSTM autoencoder model in PyTorch
class LSTM_Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTM_Autoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)
        self.hidden_dim = hidden_dim

    def forward(self, input_seq):
        """Forward pass for the LSTM Autoencoder.

        The input_seq has shape (batch_size, sequence_length, input_dim).
        """
        # Encode the input sequence; LSTM returns both encoder_output (batch_size, sequence_length, hidden_dim) and last hidden_state (n_lstm_layers, sequence_length, hidden_dim)
        encoder_output, (hidden_state, cell_state) = self.encoder(input_seq)
        last_hidden_state = hidden_state[-1]
        
        # Prepare the hidden_state for decoding by repeating it across the sequence_length; the desired shape is (batch_size, sequence_length, hidden_dim).
        batch_size = input_seq.size(1)
        repeated_hidden_state = last_hidden_state.repeat(batch_size, 1, 1)  # Shape (sequence_length, batch_size, hidden_dim)
        repeated_hidden_state = repeated_hidden_state.permute(1, 0, 2)  # Correct shape (batch_size, sequence_length, hidden_dim)
        
        # Reconstruct the input sequence from the repeated hidden_state of the encoder
        decoded_output, _ = self.decoder(repeated_hidden_state)
        
        return decoded_output
    

# Initialize the model
model = LSTM_Autoencoder(input_dim=1, hidden_dim=HPARAMS['hidden_dim']).cuda()  # Ensure input_dim matches the dataset features

# Initialize the loss function and optimizer
criterion = LOSSES[HPARAMS['loss']]
optimizer = optim.Adam(model.parameters(), lr=HPARAMS['learning_rate'])

# Initialize the information log for TensorBoard
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter(f'models/dl_runs/lstmae_experiment/{current_time}')

# Log the model architecture for TensorBoard
try:
    sample_batch = next(iter(train_loader))
    writer.add_graph(model, sample_batch.cuda())
except Exception as e:
    print("Error logging model to TensorBoard: ", e)



def train(model, data_loader, optimizer, criterion):
    model.train()  # Set the model to training mode
    total_loss = 0  # Initialize a variable to accumulate the total loss over all batches    
    for batch in data_loader:  # Loop over each batch of sequences from the DataLoader
        batch = batch.cuda()  # Move the batch to GPU for faster computation
        optimizer.zero_grad()  # Clear the gradients of all optimized parameters (important to avoid accumulation)
        output = model(batch)  # Forward pass: compute the model's output given the inputs        
        loss = criterion(output, batch)  # Compute the loss between the model's output (reconstructed input) and the target (input of the autoencoder)        
        loss.backward()  # Backward pass: compute the gradients of the loss with respect to the model's parameters        
        optimizer.step()  # Update the model's parameters using the computed gradients        
        total_loss += loss.item()  # Accumulate the loss for this batch (converted to a scalar using `.item()`)    
    return total_loss / len(data_loader)  # Return the average loss over all batches

def validate(model, data_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0  
    with torch.no_grad():  # Disable gradient computation, as we don't need to update the model
        for batch in data_loader:  
            batch = batch.cuda()  
            output = model(batch)  
            loss = criterion(output, batch)  
            total_loss += loss.item()  
    return total_loss / len(data_loader)  

# Execute training and validation over epochs
train_losses = []  # Train loss for each epoch
val_losses = []  # Validation loss for each epoch
for epoch in range(HPARAMS['num_epochs']):
    train_loss = train(model, train_loader, optimizer, criterion)
    val_loss = validate(model, val_loader, criterion)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    writer.add_scalar('Loss/Train', train_loss, epoch)
    writer.add_scalar('Loss/Val', val_loss, epoch)

# Compute final average losses
final_train_loss = np.mean(train_losses)
final_val_loss = np.mean(val_losses)
print(f'Final Training Loss: {final_train_loss:.4f}')
print(f'Final Validation Loss: {final_val_loss:.4f}')
final_metrics = {
    'final_train_loss': final_train_loss,
    'final_val_loss': final_val_loss
}
writer.add_hparams(hparam_dict=HPARAMS, metric_dict=final_metrics)

# Close the TensorBoard writer when done
writer.close()


# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()





# Detect anomalies based on reconstruction error
def detect_anomalies(model, data_loader, threshold):
    model.eval()
    reconstruction_errors = []
    with torch.no_grad():
        for sequences in data_loader:
            sequences = sequences.cuda()
            reconstructions = model(sequences)
            mse = torch.mean((sequences - reconstructions) ** 2, dim=(1, 2)).cpu().numpy()
            reconstruction_errors.extend(mse)
    return np.array(reconstruction_errors) > threshold, reconstruction_errors

# Using DataLoader to process data for anomaly detection
reconstruction_errors = []
with torch.no_grad():
    for sequences in val_loader:
        sequences = sequences.cuda()
        reconstructions = model(sequences)
        mse = torch.mean((sequences - reconstructions) ** 2, dim=(1, 2)).cpu().numpy()
        reconstruction_errors.extend(mse)

# Set a threshold for anomaly detection based on the calculated MSE
threshold = np.percentile(reconstruction_errors, 95)  # Set threshold as 95th percentile of MSE
anomalies, reconstruction_errors = detect_anomalies(model, val_loader, threshold)

# Visualize reconstruction errors and detected anomalies
plt.figure(figsize=(10, 6))
plt.plot(reconstruction_errors, label='Reconstruction Error')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.legend()
plt.show()




# # Since 'anomaly' column is missing, we skip the evaluation part based on ground truth.
# # Evaluate performance if ground truth is available
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ground_truth = pd.read_csv(data_url)['anomaly'].values[SEQ_LENGTH:][train_size:]
# print(f'Accuracy: {accuracy_score(ground_truth, anomalies):.4f}')
# print(f'Precision: {precision_score(ground_truth, anomalies):.4f}')
# print(f'Recall: {recall_score(ground_truth, anomalies):.4f}')
# print(f'F1 Score: {f1_score(ground_truth, anomalies):.4f}')
