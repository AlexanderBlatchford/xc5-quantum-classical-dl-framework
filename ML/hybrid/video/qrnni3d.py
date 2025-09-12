# learning2learn_qrnn.py
# Hybrid Quantum Recurrent Neural Network for video classification
# Works with folder-structured datasets: root/class_x/*.mp4
# Requires: tensorflow, pennylane, opencv-python, tqdm, numpy, scikit-learn (optional)

import os
import glob
import random
from tqdm import tqdm
import numpy as np
import cv2
import pennylane as qml
from pennylane import numpy as pnp
import tensorflow as tf
from sklearn.model_selection import train_test_split
import tensorflow_hub as hub


# --------------------------
# Configuration (tweak me)
# --------------------------
FRAME_SIZE = (64, 64)            # resize frames to this (HxW)
FRAMES_PER_CLIP = 8              # number of frames per clip (temporal length)
FRAME_STRIDE = 2                 # stride while sampling frames
BATCH_SIZE = 8
NUM_EPOCHS = 10
NUM_CLASSES = None               # auto-detected from data
EMBED_DIM = 32                   # classical embedding dimension for each frame
N_QUBITS = 4                      # number of qubits in QCell (hidden dim will be N_QUBITS)
Q_CIRCUIT_LAYERS = 2
LEARNING_RATE = 1e-3
DEVICE = "default.qubit"         # pennylane device
# qrnn_smallset.py
# Minimal demo: QRNN with classical CNN frame embedder
# Works on small subsets of video datasets (e.g., UCF101)

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import pennylane as qml

# -----------------------------
# CONFIG
# -----------------------------
FRAME_SIZE = (64, 64)
FRAMES_PER_CLIP = 8
EMBED_DIM = 32
N_QUBITS = 4
EPOCHS = 3
BATCH_SIZE = 2


# -----------------------------
# DATA LOADING
# -----------------------------
def load_and_preprocess_video(path, frames_per_clip=FRAMES_PER_CLIP, frame_size=FRAME_SIZE):
    """Dummy video loader — replace with real decoding."""
    # For now, just random data to simulate
    return np.random.rand(frames_per_clip, frame_size[0], frame_size[1], 3).astype("float32")


def load_dataset(data_dir, classes_subset=None, max_samples_per_class=5):
    """
    Loads dataset from folders: data_dir/class_name/*.mp4
    Args:
        classes_subset (list[str] | None): restrict to subset of classes
        max_samples_per_class (int): limit number of clips per class
    Returns:
        x (np.ndarray), y (np.ndarray), class_names (list[str])
    """
    class_names = sorted(os.listdir(data_dir))
    if classes_subset:
        class_names = [c for c in class_names if c in classes_subset]

    x, y = [], []
    for label, cname in enumerate(class_names):
        class_path = os.path.join(data_dir, cname)
        videos = [f for f in os.listdir(class_path) if f.endswith(".mp4")]
        chosen = random.sample(videos, min(max_samples_per_class, len(videos)))
        for vf in chosen:
            clip = load_and_preprocess_video(os.path.join(class_path, vf))
            x.append(clip)
            y.append(label)

    return np.array(x), np.array(y), class_names


# -----------------------------
# FRAME EMBEDDING MODEL (CNN)
# -----------------------------
def build_frame_embedding_model(input_shape=(FRAMES_PER_CLIP, FRAME_SIZE[0], FRAME_SIZE[1], 3), embed_dim=EMBED_DIM):
    inp = layers.Input(shape=input_shape)
    # Apply CNN per frame
    x = layers.TimeDistributed(layers.Conv2D(8, (3, 3), activation="relu"))(inp)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    x = layers.TimeDistributed(layers.Conv2D(16, (3, 3), activation="relu"))(x)
    x = layers.TimeDistributed(layers.Flatten())(x)
    x = layers.TimeDistributed(layers.Dense(embed_dim))(x)
    return tf.keras.Model(inp, x, name="frame_embedder")


# -----------------------------
# QUANTUM LAYER (QRNN cell)
# -----------------------------
def make_qnode(n_qubits=N_QUBITS):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="tf")
    def circuit(inputs, weights):
        # Angle encoding
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        # Variational layer
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


def build_qrnn_model(seq_len=FRAMES_PER_CLIP, embed_dim=EMBED_DIM, num_classes=2):
    qnode = make_qnode()
    weight_shapes = {"weights": (1, N_QUBITS, 3)}

    qlayer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=N_QUBITS)

    inputs = layers.Input(shape=(seq_len, embed_dim))
    x = layers.TimeDistributed(layers.Dense(N_QUBITS))(inputs)
    x = layers.TimeDistributed(qlayer)(x)
    x = layers.LSTM(16)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs, name="qrnn_model")


# -----------------------------
# TRAINING DEMO
# -----------------------------
def main():
    data_dir = "UCF101_subset"  # replace with path

    # Small subset by default
    x, y, classes = load_dataset(
        data_dir,
        classes_subset=["Basketball", "JumpingJack"],  # restrict to 2 classes
        max_samples_per_class=3                        # 3 clips per class
    )

    num_classes = len(classes)
    y = tf.keras.utils.to_categorical(y, num_classes)

    # Models
    embedder = build_frame_embedding_model()
    qrnn = build_qrnn_model(num_classes=num_classes)

    # Pipeline: frame embedder → QRNN
    inputs = layers.Input(shape=x.shape[1:])
    features = embedder(inputs)
    outputs = qrnn(features)
    model = tf.keras.Model(inputs, outputs)

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    model.fit(x, y, epochs=EPOCHS, batch_size=BATCH_SIZE)


if __name__ == "__main__":
    main()
