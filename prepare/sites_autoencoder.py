import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm


class Autoencoder(tf.keras.Model):
    def __init__(self, z_size=200):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(64, 64, 64, 2)),  # 2 input channels
            tf.keras.layers.Conv3D(64, kernel_size=(4, 4, 4), strides=(2, 2, 2), padding="same", activation="relu"),
            tf.keras.layers.Conv3D(128, kernel_size=(4, 4, 4), strides=(2, 2, 2), padding="same", activation="relu"),
            tf.keras.layers.Conv3D(256, kernel_size=(4, 4, 4), strides=(2, 2, 2), padding="same", activation="relu"),
            tf.keras.layers.Conv3D(512, kernel_size=(4, 4, 4), strides=(2, 2, 2), padding="same", activation="relu"),
            tf.keras.layers.Conv3D(z_size, kernel_size=(4, 4, 4), strides=(1, 1, 1), padding="valid", activation="tanh")
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Conv3DTranspose(512, kernel_size=(4, 4, 4), strides=(2, 2, 2), padding="same", activation="relu"),
            tf.keras.layers.Conv3DTranspose(256, kernel_size=(4, 4, 4), strides=(2, 2, 2), padding="same", activation="relu"),
            tf.keras.layers.Conv3DTranspose(128, kernel_size=(4, 4, 4), strides=(2, 2, 2), padding="same", activation="relu"),
            tf.keras.layers.Conv3DTranspose(64, kernel_size=(4, 4, 4), strides=(2, 2, 2), padding="same", activation="relu"),
            tf.keras.layers.Conv3DTranspose(2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same", activation="sigmoid")  # Match input dimensions
        ])

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return encoded, decoded

def load_data(path, batch_size):
    data_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.npy')]
    data = [np.load(f) for f in data_files]  # Keep shape (64, 64, 64, 2)
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

    @tf.function
    def train_step(batch):
        with tf.GradientTape() as tape:
            _, decoded = autoencoder(batch)
            loss = tf.reduce_mean(tf.square(batch - decoded))
        gradients = tape.gradient(loss, autoencoder.trainable_variables)
        optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables))
        return loss

    # Load data
    data_batches, data_files = load_data(sites_graph_path, batch_size)
    train_data = data_batches[:int(0.9 * len(data_batches))]
    test_data = data_batches[int(0.9 * len(data_batches)):]

    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        train_loss = 0
        for batch in tqdm(train_data, desc="Training Batches", leave=False):
            loss = train_step(batch)
            train_loss += loss.numpy()
        print(f"Epoch {epoch + 1}, Training Loss: {train_loss / len(train_data):.6f}")

        # Validation
        test_loss = 0
        for batch in tqdm(test_data, desc="Validation Batches", leave=False):
            _, decoded = autoencoder(batch)
            loss = tf.reduce_mean(tf.square(batch - decoded))
            test_loss += loss.numpy()
        print(f"Epoch {epoch + 1}, Validation Loss: {test_loss / len(test_data):.6f}")

    # Save model
    autoencoder.save(os.path.join(model_path, "autoencoder_model"))


def encode_data(autoencoder, data_files, encoded_graph_path):
    if not os.path.exists(encoded_graph_path):
        os.makedirs(encoded_graph_path)

    for data_file in tqdm(data_files, desc="Encoding Data"):
        data = np.load(data_file).reshape((1, 64, 64, 64, 2))  # Reshape for batch size
        encoded, _ = autoencoder(data)
        encoded_file = os.path.join(encoded_graph_path, os.path.basename(data_file))
        np.save(encoded_file, encoded.numpy())


# # Sample usage with provided data
# if __name__ == "__main__":
#     # Paths
#     sample_data_path = './sample_data/'
#     encoded_graph_path = './encoded_sites/'
#     model_path = './model/'

#     # Ensure sample data exists
#     os.makedirs(sample_data_path, exist_ok=True)
#     sample_npy_path = os.path.join(sample_data_path, 'sample.npy')
#     if not os.path.exists(sample_npy_path):
#         np.save(sample_npy_path, np.random.rand(64, 64, 64, 2))  # Random sample data

#     # Train autoencoder
#     train_autoencoder(sample_data_path, encoded_graph_path, model_path, epochs=10)

#     # Load trained model
#     trained_autoencoder = tf.keras.models.load_model(os.path.join(model_path, "autoencoder_model"))

#     # Encode data
#     _, data_files = load_data(sample_data_path, batch_size=1)
#     encode_data(trained_autoencoder, data_files, encoded_graph_path)
