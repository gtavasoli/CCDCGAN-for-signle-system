import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm

number_of_different_element=2

class Autoencoder(tf.keras.Model):
    def __init__(self, z_size=200):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(64, 64, 64, 2)),
            tf.keras.layers.Conv3D(64, kernel_size=(4, 4, 4), strides=(2, 2, 2), padding="same", activation="relu"),
            tf.keras.layers.Conv3D(128, kernel_size=(4, 4, 4), strides=(2, 2, 2), padding="same", activation="relu"),
            tf.keras.layers.Conv3D(256, kernel_size=(4, 4, 4), strides=(2, 2, 2), padding="same", activation="relu"),
            tf.keras.layers.Conv3D(512, kernel_size=(4, 4, 4), strides=(2, 2, 2), padding="same", activation="relu"),
            tf.keras.layers.Conv3D(z_size, kernel_size=(4, 4, 4), strides=(4, 4, 4), padding="valid", activation="tanh")
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Conv3DTranspose(512, kernel_size=(4, 4, 4), strides=(4, 4, 4), padding="valid", activation="relu"),
            tf.keras.layers.Conv3DTranspose(256, kernel_size=(4, 4, 4), strides=(2, 2, 2), padding="same", activation="relu"),
            tf.keras.layers.Conv3DTranspose(128, kernel_size=(4, 4, 4), strides=(2, 2, 2), padding="same", activation="relu"),
            tf.keras.layers.Conv3DTranspose(64, kernel_size=(4, 4, 4), strides=(2, 2, 2), padding="same", activation="relu"),
            tf.keras.layers.Conv3DTranspose(2, kernel_size=(4, 4, 4), strides=(2, 2, 2), padding="same", activation="sigmoid")
        ])

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return encoded, decoded


def load_data(path, batch_size):
    data_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.npy')]
    data = [np.load(f) for f in data_files]  # Shape: (64, 64, 64, 2)
    data_batches = [np.stack(data[i:i+batch_size]) for i in range(0, len(data), batch_size)]
    return data_batches, data_files


def train_autoencoder(sites_graph_path, encoded_graph_path, model_path, batch_size=1, z_size=200, epochs=100, learning_rate=0.0003):
    if not os.path.exists(encoded_graph_path):
        os.makedirs(encoded_graph_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Initialize autoencoder and optimizer
    autoencoder = Autoencoder(z_size=z_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Create checkpoint manager
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=autoencoder)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, model_path, max_to_keep=3)

    @tf.function
    def train_step(batch):
        with tf.GradientTape() as tape:
            encoded, decoded = autoencoder(batch)
            loss = tf.reduce_mean(tf.square(batch - decoded))
        gradients = tape.gradient(loss, autoencoder.trainable_variables)
        optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables))
        return loss, encoded

    # Load data
    data_batches, data_files = load_data(sites_graph_path, batch_size)
    train_data = data_batches[:int(0.9 * len(data_batches))]
    test_data = data_batches[int(0.9 * len(data_batches)):]

    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        train_loss = 0

        for batch_idx, batch in enumerate(tqdm(train_data, desc="Training Batches", leave=False)):
            loss, encoded = train_step(batch)
            train_loss += loss.numpy()

            # Process and save each encoded file in the batch
            for i in range(len(batch)):  # Iterate over the batch
                file_index = batch_idx * batch_size + i  # Map to the correct file index
                if file_index < len(data_files):  # Ensure index is within bounds
                    file_path = data_files[file_index]
                    encoded_path = os.path.join(encoded_graph_path, os.path.basename(file_path))

                    # Save encoded representation to file
                    padded_encoded_data = np.zeros((200, number_of_different_element), dtype=np.float32)
                    for j in range(number_of_different_element):
                        padded_encoded_data[:encoded.shape[-1], j] = encoded.numpy()[i, :]  # Extract the encoded output for this file
                    np.save(encoded_path, padded_encoded_data)

        print(f"Epoch {epoch + 1}, Training Loss: {train_loss / len(train_data):.6f}")

         # Validation
        test_loss = 0
        for batch in tqdm(test_data, desc="Validation Batches", leave=False):
            _, decoded = autoencoder(batch)
            loss = tf.reduce_mean(tf.square(batch - decoded))
            test_loss += loss.numpy()
        print(f"Epoch {epoch + 1}, Validation Loss: {test_loss / len(test_data):.6f}")

        checkpoint_manager.save()


    # Save model
    autoencoder.save(os.path.join(model_path, "autoencoder_model"), save_format="tf")
    autoencoder.save_weights(os.path.join(model_path, "sites_weights.ckpt"))

def encode_data(autoencoder, data_files, encoded_graph_path):
    if not os.path.exists(encoded_graph_path):
        os.makedirs(encoded_graph_path)

    for data_file in tqdm(data_files, desc="Encoding Data"):
        data = np.load(data_file).reshape((1, 64, 64, 64, 2))
        encoded, _ = autoencoder(data)

        padded_encoded_data = np.zeros((200, number_of_different_element), dtype=np.float32)
        for j in range(number_of_different_element):
            padded_encoded_data[:encoded.shape[-1], j] = encoded.numpy().flatten()[j]

        encoded_file = os.path.join(encoded_graph_path, os.path.basename(data_file))
        np.save(encoded_file, padded_encoded_data)

