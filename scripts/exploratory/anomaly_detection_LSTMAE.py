import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, Dense, Concatenate


# Define the LSTM autoencoder
def create_lstm_autoencoder(sequence_length, n_signals, n_metadata, latent_dim):
    """
    Creates an LSTM autoencoder with metadata integration.

    Args:
        sequence_length (int): The length of the input sequences.
        n_signals (int): The number of signals in each sequence.
        n_metadata (int): The number of metadata features.
        latent_dim (int): The dimension of the latent space.

    Returns:
        tuple: A tuple containing the autoencoder model and the encoder model.
    """
    # Input for sequence data
    sequence_input = Input(shape=(sequence_length, n_signals), name='sequence_input')

    # LSTM Encoder
    encoder_lstm = LSTM(latent_dim, activation='relu', return_sequences=False, name='encoder_lstm')
    encoded = encoder_lstm(sequence_input)

    # Metadata Input
    metadata_input = Input(shape=(n_metadata,), name='metadata_input')

    # Concatenate the latent representation with metadata
    combined = Concatenate(name='concatenate')([encoded, metadata_input])

    # Dense layer to process the combined input
    combined_dense = Dense(latent_dim, activation='relu', name='combined_dense')(combined)

    # RepeatVector to match sequence length
    repeated = RepeatVector(sequence_length)(combined_dense)

    # LSTM Decoder outputting the required shape
    sequence_output = LSTM(n_signals, activation='relu', return_sequences=True, name='decoder_lstm')(repeated)

    # Define the autoencoder model
    autoencoder = Model(inputs=[sequence_input, metadata_input], outputs=[sequence_output])
    autoencoder.compile(optimizer='adam', loss='mse')

    # Return both the autoencoder and the encoder model
    encoder_model = Model(inputs=[sequence_input, metadata_input], outputs=encoded)

    return autoencoder, encoder_model


# Generate synthetic training data
def generate_synthetic_data(n_samples, sequence_length, n_signals, n_metadata):
    """
    Generates synthetic training data.

    Args:
        n_samples (int): Number of samples to generate.
        sequence_length (int): The length of each sequence.
        n_signals (int): The number of signals in each sequence.
        n_metadata (int): The number of metadata features.

    Returns:
        tuple: A tuple containing the generated sequences and metadata.
    """
    sequences = np.random.rand(n_samples, sequence_length, n_signals)
    metadata = np.random.rand(n_samples, n_metadata)
    return sequences, metadata


# Generate synthetic test data with anomalies
def generate_test_data_with_anomalies(n_samples, sequence_length, n_signals, n_metadata, anomaly_fraction=0.1):
    """
    Generates synthetic test data with anomalies.

    Args:
        n_samples (int): Number of samples to generate.
        sequence_length (int): The length of each sequence.
        n_signals (int): The number of signals in each sequence.
        n_metadata (int): The number of metadata features.
        anomaly_fraction (float): The fraction of samples to introduce anomalies in.

    Returns:
        tuple: A tuple containing the generated sequences, metadata, and indices of anomalies.
    """
    sequences, metadata = generate_synthetic_data(n_samples, sequence_length, n_signals, n_metadata)

    # Introduce anomalies
    n_anomalies = int(n_samples * anomaly_fraction)
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    sequences[anomaly_indices] += np.random.normal(0, 1, size=(n_anomalies, sequence_length, n_signals))

    return sequences, metadata, anomaly_indices


def plot_latent_space(latent_representations, true_anomalies, detected_anomalies):
    """
    Plots the t-SNE of the latent space with anomalies highlighted.

    Args:
        latent_representations (np.ndarray): The latent representations of the sequences.
        true_anomalies (np.ndarray): Indices of the true anomalies.
        detected_anomalies (np.ndarray): Boolean array indicating detected anomalies.

    Returns:
        None
    """
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    latent_2d = tsne.fit_transform(latent_representations)

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c='blue', label='Normal')
    plt.scatter(latent_2d[true_anomalies, 0], latent_2d[true_anomalies, 1], c='yellow', label='True Anomaly')
    plt.scatter(latent_2d[detected_anomalies, 0], latent_2d[detected_anomalies, 1], edgecolors='r', facecolors='none', label='Detected Anomaly')
    plt.title('t-SNE of Latent Representations - Anomalies in Yellow, Detected Anomalies Circled')
    plt.legend()
    plt.show()


def main():
    # Parameters
    sequence_length = 50
    n_signals = 5
    n_metadata = 3
    latent_dim = 64
    n_train_samples = 1000
    n_test_samples = 100

    # Generate training data
    train_sequences, train_metadata = generate_synthetic_data(n_train_samples, sequence_length, n_signals, n_metadata)

    # Generate test data with anomalies
    test_sequences, test_metadata, true_anomalies = generate_test_data_with_anomalies(n_test_samples, sequence_length, n_signals, n_metadata)

    # Create the autoencoder
    autoencoder, encoder_model = create_lstm_autoencoder(sequence_length, n_signals, n_metadata, latent_dim)

    # Train the autoencoder
    autoencoder.fit([train_sequences, train_metadata], train_sequences, epochs=50, batch_size=32)

    # Perform anomaly detection
    reconstructed_sequences = autoencoder.predict([test_sequences, test_metadata])
    reconstruction_errors = np.linalg.norm(test_sequences - reconstructed_sequences, axis=(1, 2))

    # Define a threshold for anomaly detection
    threshold = np.mean(reconstruction_errors) + 2 * np.std(reconstruction_errors)

    # Detect anomalies
    detected_anomalies = reconstruction_errors > threshold

    # Evaluate detection
    true_positive = np.sum(detected_anomalies[true_anomalies])
    false_positive = np.sum(detected_anomalies) - true_positive
    false_negative = len(true_anomalies) - true_positive

    # Print results
    print("Reconstruction Errors:", reconstruction_errors)
    print("Detected Anomalies:", detected_anomalies)
    print("Number of anomalies detected:", np.sum(detected_anomalies))
    print("True Positives:", true_positive)
    print("False Positives:", false_positive)
    print("False Negatives:", false_negative)

    # Get latent representations
    latent_representations = encoder_model.predict([test_sequences, test_metadata])

    # Plot latent space
    plot_latent_space(latent_representations, true_anomalies, detected_anomalies)


if __name__ == "__main__":
    main()
