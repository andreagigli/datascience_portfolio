import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense


"""Step 1: Load the dataset
We'll use the NAB (Numenta Anomaly Benchmark) dataset.
"""
# Load a synthetic dataset
data_url = "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv"
data = pd.read_csv(data_url)
data['value'] = MinMaxScaler().fit_transform(data['value'].values.reshape(-1, 1))
data = data['value'].values

# Visualize the data
plt.plot(data)
plt.show(block=True)


""" Step 2: Preprocess the data
Create sequences for LSTM input.
"""
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


"""Step 3: Define the LSTM autoencoder model
python
"""
input_dim = train_sequences.shape[2]
timesteps = train_sequences.shape[1]

inputs = Input(shape=(timesteps, input_dim))

# Encoder
encoded = LSTM(64)(inputs)

# Decoder
decoded = RepeatVector(timesteps)(encoded)
decoded = LSTM(64, return_sequences=True)(decoded)
decoded = TimeDistributed(Dense(input_dim))(decoded)

# Autoencoder model
autoencoder = Model(inputs, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.summary()


"""Step 4: Train the model"""
history = autoencoder.fit(train_sequences, train_sequences, epochs=20, batch_size=32, validation_split=0.1)


"""Step 5: Detect anomalies based on reconstruction error"""
def detect_anomalies(model, sequences, threshold):
    reconstructions = model.predict(sequences)
    mse = np.mean(np.power(sequences - reconstructions, 2), axis=(1, 2))
    return mse > threshold, mse

# Calculate reconstruction loss on the test set
reconstructions = autoencoder.predict(test_sequences)
mse = np.mean(np.power(test_sequences - reconstructions, 2), axis=(1, 2))

# Set a threshold for anomaly detection
threshold = np.percentile(mse, 95)  # e.g., 95th percentile of training loss
anomalies, reconstruction_errors = detect_anomalies(autoencoder, test_sequences, threshold)

# Visualize reconstruction errors and detected anomalies
plt.figure(figsize=(10, 6))
plt.plot(reconstruction_errors, label='Reconstruction Error')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.legend()
plt.show()


"""Step 6: Evaluate performance if ground truth is available"""
# Assuming ground truth is available in test dataset
ground_truth = pd.read_csv(data_url)['anomaly'].values[SEQ_LENGTH:][train_size:]
print(f'Accuracy: {accuracy_score(ground_truth, anomalies):.4f}')
print(f'Precision: {precision_score(ground_truth, anomalies):.4f}')
print(f'Recall: {recall_score(ground_truth, anomalies):.4f}')
print(f'F1 Score: {f1_score(ground_truth, anomalies):.4f}')



