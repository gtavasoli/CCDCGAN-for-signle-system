import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import prepare.data_transformation as dt
import time

# Hyperparameters
BATCH_SIZE = 1
Z_SIZE = 25
LEARNING_RATE = 0.0003
EPOCHS = 51 #Original code 201
REG_L2 = 0.0e-6
INPUT_SHAPE = (32, 32, 32, 1)

# Xavier initializer
initializer = tf.keras.initializers.GlorotUniform()

# Threshold function
def threshold(x, val=0.5):
    x = tf.clip_by_value(x, 0.5, 0.5001) - 0.5
    x = tf.minimum(x * 10000, 1)
    return x

# Define encoder
def build_encoder(weights):
    def encoder_fn(inputs, training=True):
        strides = [1, 2, 2, 2, 1]
        x = tf.nn.conv3d(inputs, weights['wae1'], strides=strides, padding="SAME")
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = tf.nn.conv3d(x, weights['wae2'], strides=strides, padding="SAME")
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = tf.nn.conv3d(x, weights['wae3'], strides=strides, padding="SAME")
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = tf.nn.conv3d(x, weights['wae4'], strides=[1, 1, 1, 1, 1], padding="VALID")
        return tf.nn.tanh(x)
    return encoder_fn

# Define decoder
def build_decoder(weights):
    def decoder_fn(inputs, training=True):
        strides = [1, 2, 2, 2, 1]
        x = tf.nn.conv3d_transpose(inputs, weights['wg1'], [BATCH_SIZE, 4, 4, 4, 64], strides=[1, 1, 1, 1, 1], padding="VALID")
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = tf.nn.conv3d_transpose(x, weights['wg2'], [BATCH_SIZE, 8, 8, 8, 64], strides=strides, padding="SAME")
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = tf.nn.conv3d_transpose(x, weights['wg3'], [BATCH_SIZE, 16, 16, 16, 64], strides=strides, padding="SAME")
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = tf.nn.conv3d_transpose(x, weights['wg4'], [BATCH_SIZE, 32, 32, 32, 1], strides=strides, padding="SAME")
        return tf.nn.sigmoid(x)
    return decoder_fn

# Initialize weights
def initialise_weights():
    weights = {
        'wg1': tf.Variable(initializer([4, 4, 4, 64, Z_SIZE]), name='wg1'),
        'wg2': tf.Variable(initializer([4, 4, 4, 64, 64]), name='wg2'),
        'wg3': tf.Variable(initializer([4, 4, 4, 64, 64]), name='wg3'),
        'wg4': tf.Variable(initializer([4, 4, 4, 1, 64]), name='wg4'),
        'wae1': tf.Variable(initializer([4, 4, 4, 1, 64]), name='wae1'),
        'wae2': tf.Variable(initializer([4, 4, 4, 64, 64]), name='wae2'),
        'wae3': tf.Variable(initializer([4, 4, 4, 64, 64]), name='wae3'),
        'wae4': tf.Variable(initializer([4, 4, 4, 64, Z_SIZE]), name='wae4'),
    }
    return weights

# Train the autoencoder
def train_autoencoder(lattice_graph_path, encoded_graph_path, model_path):
    os.makedirs(encoded_graph_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    weights = initialise_weights()
    encoder = build_encoder(weights)
    decoder = build_decoder(weights)

    x_vector = tf.Variable(tf.zeros([BATCH_SIZE, 32, 32, 32, 1]), trainable=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    train_list, test_list = dt.train_test_split(path=lattice_graph_path, split_ratio=0.1)

    min_test_loss = float('inf')
    for epoch in range(EPOCHS):
        start_time = time.time()
        train_loss = 0

        print(f"Starting Epoch {epoch+1}/{EPOCHS}")
        with tqdm(total=len(train_list), desc="Training", unit="batch") as pbar:
            for batch in dt.get_batch_name_list(train_list, batch_size=BATCH_SIZE):
                inputs_batch = np.load(os.path.join(lattice_graph_path, batch + '.npy')).reshape(BATCH_SIZE, 32, 32, 32, 1)
                x_vector.assign(inputs_batch)

                with tf.GradientTape() as tape:
                    encoded = encoder(x_vector, training=True)
                    decoded = decoder(encoded, training=True)
                    decoded = threshold(decoded)
                    mse_loss = tf.reduce_mean(tf.square(x_vector - decoded))
                    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in weights.values()])
                    total_loss = mse_loss + REG_L2 * l2_loss

                gradients = tape.gradient(total_loss, weights.values())
                optimizer.apply_gradients(zip(gradients, weights.values()))
                train_loss += total_loss.numpy()
                pbar.update(1)

        test_loss = 0
        with tqdm(total=len(test_list), desc="Testing", unit="batch") as pbar:
            for batch in dt.get_batch_name_list(test_list, batch_size=BATCH_SIZE):
                inputs_batch = np.load(os.path.join(lattice_graph_path, batch + '.npy')).reshape(BATCH_SIZE, 32, 32, 32, 1)
                x_vector.assign(inputs_batch)

                encoded = encoder(x_vector, training=False)
                decoded = decoder(encoded, training=False)
                decoded = threshold(decoded)
                mse_loss = tf.reduce_mean(tf.square(x_vector - decoded))
                test_loss += mse_loss.numpy()
                pbar.update(1)

        if test_loss < min_test_loss:
            min_test_loss = test_loss
            print("Saving best model...")
            for name in train_list + test_list:
                inputs = np.load(os.path.join(lattice_graph_path, name + '.npy')).reshape(BATCH_SIZE, 32, 32, 32, 1)
                x_vector.assign(inputs)
                encoded = encoder(x_vector, training=False).numpy().reshape(Z_SIZE)
                np.save(os.path.join(encoded_graph_path, name + '.npy'), encoded)

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f} | Time: {epoch_time:.2f}s")

# Restore and decode lattices
def lattice_restorer(encoded_graph_path, decoded_graph_path, model_path):
    os.makedirs(decoded_graph_path, exist_ok=True)

    weights = initialise_weights()
    encoder = build_encoder(weights)
    decoder = build_decoder(weights)

    print("Starting Decoding...")
    with tqdm(total=len(os.listdir(encoded_graph_path)), desc="Decoding", unit="file") as pbar:
        for filename in os.listdir(encoded_graph_path):
            if filename.endswith(".npy"):
                encoded_lattice = np.load(os.path.join(encoded_graph_path, filename)).reshape(BATCH_SIZE, 1, 1, 1, Z_SIZE)
                decoded_lattice = decoder(encoded_lattice, training=False).numpy().reshape(32, 32, 32)
                np.save(os.path.join(decoded_graph_path, filename), decoded_lattice)
                pbar.update(1)
