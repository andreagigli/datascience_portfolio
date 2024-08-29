import dask.dataframe as dd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter





""" Explanation of the implemented data handling process:
-----------------------------------------------------
This implementation efficiently handles large time series datasets by splitting the dataset, 
loading sequences one at a time, and performing incremental scaling during the batching process. 
The key steps are:

1. **Data Splitting:**
- The dataset is split into training and validation subsets using `torch.utils.data.Subset`. 
- This splitting is index-based, meaning the data isn't loaded until accessed via the `DataLoader`.

2. **Data Loading:** 
- The `TimeSeriesDataset` class, through its `__getitem__` method, loads one sequence at a time from a CSV file. 
- Each sequence is a fixed-length chunk of time series data.

3. **Batching:**
- The `DataLoader` handles batching by calling the `__getitem__` method of `TimeSeriesDataset` multiple times 
    to collect sequences into batches.
- A custom collate function (`collate_and_scale`) scales each batch using an `IncrementalScaler`.

4. **Scaling:**
- The `IncrementalScaler` class scales the data either by fitting and transforming or by just transforming 
    based on the `keep_fitting` flag. This allows flexibility in controlling the scaling behavior during training and validation.
-----------------------------------------------------
"""

# Define the TimeSeriesDataset class for online loading and preprocessing using Pandas
class TimeSeriesDataset(Dataset):
    """
    A PyTorch Dataset class for loading and preprocessing time series data efficiently using Pandas.

    Attributes:
        file_path (str): Path to the CSV file containing the time series data.
        sequence_length (int): The length of each sequence to be returned by the dataset.
        padding_value (float): The value to use for padding sequences that are shorter than the sequence_length.
        data_length (int): The total number of rows in the dataset, used to calculate the number of sequences.
    """

    def __init__(self, file_path, sequence_length, padding_value=0):
        """
        Args:
            file_path (str): Path to the CSV file containing the time series data.
            sequence_length (int): The length of each sequence to be returned by the dataset.
            padding_value (float): The value to use for padding sequences that are shorter than the sequence_length.
        """
        self.file_path = file_path
        self.sequence_length = sequence_length
        self.padding_value = padding_value

        # Compute the number of rows in the CSV file without explicitly loading the dataset
        self.data_length = sum(1 for _ in open(self.file_path)) - 1  # Subtract 1 for header

    def __len__(self):
        """Returns the number of sequences that can be generated from the data."""
        return self.data_length - self.sequence_length + 1

    def __getitem__(self, index):
        """
        Loads a chunk of data from the CSV file and returns it as a tensor.

        Args:
            index (int): The index of the sequence to retrieve.

        Returns:
            torch.Tensor: The loaded data as a tensor.
        """
        # Load the relevant chunk of data using pandas
        data_chunk = pd.read_csv(self.file_path, skiprows=range(1, index + 1), nrows=self.sequence_length, header=0)
        values = data_chunk.loc[:, "value"].astype(float).values.reshape(-1, 1)
        
        # Convert the values to a PyTorch tensor
        data_tensor = torch.tensor(values, dtype=torch.float32)
        
        return data_tensor
    

class IncrementalScaler:
    """
    A class to handle scaling of data in an incremental manner, useful for online learning scenarios.

    Attributes:
        scaler (MinMaxScaler): The scaler object used for scaling the data.
        scaler_fitted (bool): Indicates whether the scaler has been fitted.
        keep_fitting (bool): Controls whether the scaler should continue fitting on new data.
    """

    def __init__(self, scaler=None, keep_fitting=True):
        """
        Args:
            scaler (MinMaxScaler, optional): The scaler object used for scaling the data. Defaults to MinMaxScaler().
            keep_fitting (bool, optional): Controls whether the scaler should continue fitting on new data. Defaults to True.
        """
        self.scaler = scaler if scaler is not None else MinMaxScaler()
        self.scaler_fitted = False
        self.keep_fitting = keep_fitting  # Whether to keep fitting the scaler

    def fit_transform(self, data):
        """
        Fits the scaler to the data if necessary, then transforms the data.

        Args:
            data (np.ndarray): The data to be scaled.

        Returns:
            np.ndarray: The scaled data.
        """
        if not self.scaler_fitted or self.keep_fitting:
            # Fit and transform the data if the scaler is not fitted yet or if keep_fitting is True
            scaled_data = self.scaler.fit_transform(data)
            self.scaler_fitted = True
        else:
            # Only transform if the scaler is already fitted and keep_fitting is False
            scaled_data = self.scaler.transform(data)
        return scaled_data
    
    def set_keep_fitting(self, keep_fitting):
        """
        Setter method to update the keep_fitting flag.

        Args:
            keep_fitting (bool): The new value for the keep_fitting flag.
        """
        self.keep_fitting = keep_fitting

class CollateFnWrapper:
    def __init__(self, scaler):
        self.scaler = scaler

    def __call__(self, batch):
        return collate_and_scale(batch, self.scaler)


def collate_and_scale(batch, incremental_scaler):
    """
    Custom collate function that stacks individual samples into a batch and scales them using the IncrementalScaler.

    Args:
        batch (list of torch.Tensor): The list of samples to collate into a batch.
        incremental_scaler (IncrementalScaler): The scaler to use for scaling the batch.

    Returns:
        torch.Tensor: The scaled batch as a tensor.
    """
    # Stack the individual samples into a batch
    batch_tensor = torch.stack(batch)
    
    # Use the incremental scaler to scale the batch
    scaled_batch = incremental_scaler.fit_transform(batch_tensor.numpy().reshape(-1, 1)).reshape(batch_tensor.shape)
    
    # Convert back to a tensor
    return torch.tensor(scaled_batch, dtype=torch.float32)

# Define the LSTM autoencoder model in PyTorch
# class LSTM_Autoencoder(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super(LSTM_Autoencoder, self).__init__()
#         self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
#         self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)
#         self.hidden_dim = hidden_dim

#     def forward(self, input_seq):
#         """Forward pass for the LSTM Autoencoder.

#         The input_seq has shape (batch_size, sequence_length, input_dim).
#         """
#         # Encode the input sequence; LSTM returns both encoder_output (batch_size, sequence_length, hidden_dim) and last hidden_state (n_lstm_layers, sequence_length, hidden_dim)
#         encoder_output, (hidden_state, cell_state) = self.encoder(input_seq)
#         last_hidden_state = hidden_state[-1]
        
#         # Prepare the hidden_state for decoding by repeating it across the sequence_length; the desired shape is (batch_size, sequence_length, hidden_dim).
#         batch_size = input_seq.size(1)
#         repeated_hidden_state = last_hidden_state.repeat(batch_size, 1, 1)  # Shape (sequence_length, batch_size, hidden_dim)
#         repeated_hidden_state = repeated_hidden_state.permute(1, 0, 2)  # Correct shape (batch_size, sequence_length, hidden_dim)
        
#         # Reconstruct the input sequence from the repeated hidden_state of the encoder
#         decoded_output, _ = self.decoder(repeated_hidden_state)
        
#         return decoded_output

class LSTM_Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_length, num_layers=2):
        super(LSTM_Autoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers=num_layers, batch_first=True)
        
        # Store sequence length
        self.seq_length = seq_length

    def forward(self, x):
        # Encoding
        _, (hidden_state, _) = self.encoder(x)
        
        # Repeat hidden state across the sequence length and permute
        hidden_state = hidden_state[-1].unsqueeze(1).repeat(1, self.seq_length, 1)
        
        # Decoding
        decoded_output, _ = self.decoder(hidden_state)
        
        return decoded_output


def main():
    # Define hyperparameters
    LOSSES = {
        "mse": nn.MSELoss(), 
    }

    HPARAMS = {
        'seq_length': 50,
        'n_lstm_layers': 1,
        'learning_rate': 0.01,
        'hidden_dim': 5,
        'batch_size': 16,
        'num_epochs': 20,
        'loss': 'mse',
    }

    data_path = "./data/external/nyctaxidb/nyctaxidb.csv"

    
    
    # # Create the dataset using Pandas for online loading and preprocessing    # TODO: UNCOMMENT FOR ONLINE DATA PROCESSING
    # dataset = TimeSeriesDataset(data_path, HPARAMS['seq_length'])

    # # Split the dataset
    # train_size = int(0.8 * len(dataset))
    # test_size = len(dataset) - train_size

    # train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
    # test_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

    # # Initialize the IncrementalScaler
    # incremental_scaler = IncrementalScaler()

    # # Create a collate function wrapper
    # collate_fn = CollateFnWrapper(incremental_scaler)

    # # DataLoader for batching, no shuffling for time series
    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False,
    #                         collate_fn=collate_fn,
    #                         pin_memory=True)

    # val_loader = DataLoader(test_dataset, batch_size=64, shuffle=False,
    #                         collate_fn=collate_fn,
    #                         pin_memory=True)



    # DEBUG: BATCH DATA PROCESSING FOR FAST PROTOTYPING
    # Step 1: Load the dataset using pandas
    data = pd.read_csv(data_path)
    values = data.loc[:, "value"].astype(float).values.reshape(-1, 1)

    # Step 2: Apply scaling to the dataset
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(values)

    # Step 3: Split the dataset into training and testing datasets
    train_size = int(0.7 * len(scaled_values))
    train_values = scaled_values[:train_size]
    test_values = scaled_values[train_size:]

    # Step 4: Create sequences using a sliding window
    def create_sequences(data, seq_length):
        sequences = []
        for i in range(len(data) - seq_length + 1):
            sequences.append(data[i:i + seq_length])
        return np.array(sequences)

    train_sequences = create_sequences(train_values, HPARAMS['seq_length'])
    test_sequences = create_sequences(test_values, HPARAMS['seq_length'])

    # Step 5: Convert sequences to PyTorch tensors
    train_sequences = torch.tensor(train_sequences, dtype=torch.float32)
    test_sequences = torch.tensor(test_sequences, dtype=torch.float32)

    # Step 6: Create DataLoader instances for batching
    train_loader = DataLoader(train_sequences, batch_size=HPARAMS['batch_size'], shuffle=False)
    val_loader = DataLoader(test_sequences, batch_size=HPARAMS['batch_size'], shuffle=False)



    
    # Initialize the model
    model = LSTM_Autoencoder(
        input_dim=1, 
        hidden_dim=HPARAMS['hidden_dim'], 
        seq_length=HPARAMS['seq_length'],  # Pass the sequence length here
        num_layers=HPARAMS['n_lstm_layers'],
    ).cuda()

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


    def train(model, data_loader, optimizer, criterion, writer, epoch):
        model.train()  # Set the model to training mode
        total_loss = 0  # Initialize a variable to accumulate the total loss over all batches    
        for batch in data_loader:  # Loop over each batch of sequences from the DataLoader
            batch = batch.cuda()  # Move the batch to GPU for faster computation
            optimizer.zero_grad()  # Clear the gradients of all optimized parameters (important to avoid accumulation)
            output = model(batch)  # Forward pass: compute the model's output given the inputs        
            loss = criterion(output, batch)  # Compute the loss between the model's output (reconstructed input) and the target (input of the autoencoder)        
            loss.backward()  # Backward pass: compute the gradients of the loss with respect to the model's parameters
            
            # Log the gradients to TensorBoard
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    writer.add_histogram(f'{name}_grad', param.grad, epoch)
                    writer.add_scalar(f'{name}_grad_norm', param.grad.norm().item(), epoch)
            
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
        # incremental_scaler.set_keep_fitting(True)  # Enable further fitting  # TODO: UNCOMMENT FOR ONLINE DATA PROCESSING
        train_loss = train(model, train_loader, optimizer, criterion, writer, epoch)
        # incremental_scaler.set_keep_fitting(False)  # Disable further fitting  # TODO: UNCOMMENT FOR ONLINE DATA PROCESSING
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



    # 1. Mean Baseline
    def mean_baseline(val_loader):
        total_loss = 0
        for batch in val_loader:
            batch = batch.cuda()
            mean_reconstructed = torch.mean(batch, dim=1, keepdim=True).repeat(1, batch.size(1), 1)
            loss = criterion(mean_reconstructed, batch)
            total_loss += loss.item()
        return total_loss / len(val_loader)

    mean_loss = mean_baseline(val_loader)
    print(f'Mean Baseline Loss: {mean_loss:.4f}')



    # 2. PCA Baseline using sklearn
    def pca_baseline(train_loader, val_loader, n_components):
        # Flatten each sequence in the training data and fit PCA
        train_flat = []
        for batch in train_loader:
            # Remove the last dimension (feature) and apply PCA along the sequence length
            batch = batch.squeeze(-1).cpu().numpy()  # Shape: (batch_size, seq_length)
            train_flat.append(batch)
        train_flat = np.concatenate(train_flat, axis=0)  # Shape: (batch_size * n_batches, seq_length)

        # Fit PCA
        pca_model = PCA(n_components=n_components)
        pca_model.fit(train_flat)  # Fit PCA to the sequences

        # Reconstruct the validation data using the PCA model
        total_loss = 0
        for batch in val_loader:
            original_shape = batch.shape
            batch_np = batch.squeeze(-1).cpu().numpy()  # Convert batch to NumPy for PCA, shape: (batch_size, seq_length)
            transformed = pca_model.transform(batch_np)  # Compress the sequence length
            reconstructed = pca_model.inverse_transform(transformed)  # Reconstruct the sequence            
            reconstructed = torch.tensor(reconstructed, dtype=torch.float32).unsqueeze(-1).cuda()  # Shape: (batch_size, seq_length, 1)
            loss = criterion(reconstructed, batch.cuda())  # Compare with the original PyTorch tensor batch
            total_loss += loss.item()
        
        return total_loss / len(val_loader)

    # PCA Baseline Loss Calculation
    n_components_pca = HPARAMS['seq_length']//10
    pca_loss = pca_baseline(train_loader, val_loader, n_components=n_components_pca)
    print(f'PCA Baseline Loss ({n_components_pca} components): {pca_loss:.4f}')



    # DEBUG: PLOT AE-RECONSTRUCTED SEQUENCES
    def plot_reconstructions(model, data_loader, num_examples=5):
        model.eval()
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.cuda()
                reconstructions = model(batch).cpu().numpy()
                batch = batch.cpu().numpy()

                for i in range(min(num_examples, batch.shape[0])):
                    plt.figure(figsize=(10, 4))
                    plt.plot(batch[i], label='Original')
                    plt.plot(reconstructions[i], label='Reconstructed')
                    plt.legend()
                    plt.title(f'Sample {i + 1}')
                break
            plt.pause(1)

    plot_reconstructions(model, val_loader, num_examples=5)


    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    def visualize_latent_space(encoder, data_loader, method='pca'):
        encoder.eval()
        latent_vectors = []
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.cuda()
                latent_vec = encoder(batch).cpu().numpy()
                latent_vectors.append(latent_vec)
            latent_vectors = np.concatenate(latent_vectors, axis=0)

            if method == 'pca':
                pca = PCA(n_components=2)
                reduced_latent = pca.fit_transform(latent_vectors)
            elif method == 'tsne':
                tsne = TSNE(n_components=2)
                reduced_latent = tsne.fit_transform(latent_vectors)

            plt.scatter(reduced_latent[:, 0], reduced_latent[:, 1], s=2)
            plt.title(f'Latent Space ({method.upper()})')
            plt.show()

    visualize_latent_space(model.encoder, val_loader, method='pca')




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

    # ground_truth = pd.read_csv(data_url)['anomaly'].values[HPARAMS['seq_length']:][train_size:]
    # print(f'Accuracy: {accuracy_score(ground_truth, anomalies):.4f}')
    # print(f'Precision: {precision_score(ground_truth, anomalies):.4f}')
    # print(f'Recall: {recall_score(ground_truth, anomalies):.4f}')
    # print(f'F1 Score: {f1_score(ground_truth, anomalies):.4f}')


if __name__=="__main__":
    mp.set_start_method('spawn')
    main()
