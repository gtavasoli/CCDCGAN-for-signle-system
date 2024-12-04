import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm 
import time

# Hyperparameters
BATCH_SIZE = 16
Z_SIZE = 25
LEARNING_RATE = 0.0003 * (BATCH_SIZE / 4)
EPOCHS = 201
REG_L2 = 0.0e-6
INPUT_SHAPE = (32, 32, 32, 1)

# Xavier initializer
xavier_init = tf.keras.initializers.GlorotUniform()

# Define encoder
def build_encoder():
    inputs = tf.keras.Input(shape=INPUT_SHAPE)
    x = tf.keras.layers.Conv3D(64, kernel_size=4, strides=2, padding="same", activation=tf.nn.leaky_relu,
                               kernel_initializer=xavier_init)(inputs)
    x = tf.keras.layers.Conv3D(64, kernel_size=4, strides=2, padding="same", activation=tf.nn.leaky_relu,
                               kernel_initializer=xavier_init)(x)
    x = tf.keras.layers.Conv3D(64, kernel_size=4, strides=2, padding="same", activation=tf.nn.leaky_relu,
                               kernel_initializer=xavier_init)(x)
    outputs = tf.keras.layers.Conv3D(Z_SIZE, kernel_size=4, strides=1, padding="valid", activation=tf.nn.tanh,
                                     kernel_initializer=xavier_init)(x)
    return tf.keras.Model(inputs, outputs, name="encoder")

# Define decoder
def build_decoder():
    inputs = tf.keras.Input(shape=(1, 1, 1, Z_SIZE))
    x = tf.keras.layers.Conv3DTranspose(64, kernel_size=4, strides=1, padding="valid", activation=tf.nn.leaky_relu,
                                        kernel_initializer=xavier_init)(inputs)
    x = tf.keras.layers.Conv3DTranspose(64, kernel_size=4, strides=2, padding="same", activation=tf.nn.leaky_relu,
                                        kernel_initializer=xavier_init)(x)
    x = tf.keras.layers.Conv3DTranspose(64, kernel_size=4, strides=2, padding="same", activation=tf.nn.leaky_relu,
                                        kernel_initializer=xavier_init)(x)
    outputs = tf.keras.layers.Conv3DTranspose(1, kernel_size=4, strides=2, padding="same", activation=tf.nn.sigmoid,
                                              kernel_initializer=xavier_init)(x)
    return tf.keras.Model(inputs, outputs, name="decoder")

# Threshold function
def threshold(x, val=0.5):
    x = tf.clip_by_value(x, 0.5, 0.5001) - 0.5
    x = tf.minimum(x * 10000, 1)
    return x

# Load and preprocess data
def load_data(path, batch_size=BATCH_SIZE):
    def process_file(filename):
        data = np.load(filename.numpy())
        return tf.convert_to_tensor(data.reshape(INPUT_SHAPE), dtype=tf.float32)

    dataset = tf.data.Dataset.list_files(os.path.join(path, "*.npy"))
    dataset = dataset.map(lambda x: tf.py_function(func=process_file, inp=[x], Tout=tf.float32))
    dataset = dataset.batch(batch_size).shuffle(buffer_size=100)
    return dataset

# Train the autoencoder
def train_autoencoder(lattice_graph_path, encoded_graph_path, model_path):
    os.makedirs(encoded_graph_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    encoder = build_encoder()
    decoder = build_decoder()

    inputs = tf.keras.Input(shape=INPUT_SHAPE)
    encoded = encoder(inputs)
    decoded = decoder(encoded)
    autoencoder = tf.keras.Model(inputs, decoded, name="autoencoder")

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    mse_loss = tf.keras.losses.MeanSquaredError()

    train_dataset = load_data(lattice_graph_path)

    print("Starting training...")
    for epoch in range(EPOCHS):
        start_time = time.time()  # Start timing the epoch
        train_loss = 0

        with tqdm(total=len(list(train_dataset)), desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch") as pbar:
            for step, x_batch in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    encoded = encoder(x_batch)
                    decoded = decoder(encoded)
                    decoded = threshold(decoded)
                    loss = mse_loss(x_batch, decoded)
                    train_loss += loss.numpy()

                gradients = tape.gradient(loss, autoencoder.trainable_variables)
                optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables))

                pbar.update(1)  # Update the progress bar
                pbar.set_postfix({"Batch Loss": f"{loss.numpy():.6f}"})  # Update the postfix with loss

        epoch_loss = train_loss / len(list(train_dataset))
        elapsed_time = time.time() - start_time  # Calculate epoch duration
        print(f"Epoch {epoch+1}/{EPOCHS}: Average Loss = {epoch_loss:.6f}, Time Taken = {elapsed_time:.2f}s")

        # Save the model
        if (epoch + 1) % 10 == 0:
            autoencoder.save_weights(os.path.join(model_path, "lattice_autoencoder.weights.h5"))
            print(f"Model saved at epoch {epoch+1}.")

    print("Training completed. Starting encoding...")

    lattice_files = [f for f in os.listdir(lattice_graph_path) if f.endswith(".npy")]
    with tqdm(total=len(lattice_files), desc="Encoding lattices") as pbar:
        for filename in lattice_files:
            lattice = np.load(os.path.join(lattice_graph_path, filename))
            lattice = lattice.reshape((1,) + INPUT_SHAPE)
            encoded_lattice = encoder.predict(lattice).reshape(Z_SIZE)
            np.save(os.path.join(encoded_graph_path, filename), encoded_lattice)
            
            # Update progress bar with the current file name
            pbar.set_description(f"Encoding: {filename}")
            pbar.update(1)

# Restore and decode lattices
def lattice_restorer(encoded_graph_path, decoded_graph_path, model_path):
    os.makedirs(decoded_graph_path, exist_ok=True)

    encoder = build_encoder()
    decoder = build_decoder()

    print("Restoring models...")
    encoder.load_weights(os.path.join(model_path, "lattice_autoencoder.h5"))
    decoder.load_weights(os.path.join(model_path, "lattice_autoencoder.h5"))
    print("Models restored successfully.")

    print("Starting decoding...")
    for filename in os.listdir(encoded_graph_path):
        if filename.endswith(".npy"):
            encoded = np.load(os.path.join(encoded_graph_path, filename)).reshape((1, 1, 1, 1, Z_SIZE))
            decoded = decoder.predict(encoded).reshape((32, 32, 32))
            np.save(os.path.join(decoded_graph_path, filename), decoded)
            print(f"Decoded lattice saved: {filename}")
