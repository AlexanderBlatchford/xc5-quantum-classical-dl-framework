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

# --------------------------
# Utilities: video -> frames
# --------------------------
def sample_frames_from_video(video_path, frames_per_clip=FRAMES_PER_CLIP, stride=FRAME_STRIDE):
    """Read video and sample `frames_per_clip` frames with given stride.
       Returns list of frames as HxWx3 uint8 arrays."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    success, frame = cap.read()
    while success:
        frames.append(frame)
        success, frame = cap.read()
    cap.release()
    if len(frames) == 0:
        return None
    # if too short, loop frames
    if len(frames) < frames_per_clip * stride:
        # simple tiling
        while len(frames) < frames_per_clip * stride:
            frames = frames + frames
    # sample evenly with stride
    start = 0
    sampled = []
    idx = start
    while len(sampled) < frames_per_clip:
        sampled.append(frames[int(idx) % len(frames)])
        idx += stride
    # resize and convert BGR -> RGB, normalize to [0,1]
    processed = [cv2.cvtColor(cv2.resize(f, FRAME_SIZE), cv2.COLOR_BGR2RGB) / 255.0 for f in sampled]
    return np.stack(processed).astype(np.float32)  # shape (T, H, W, 3)

# --------------------------
# Dataset loader (folder structure)
# --------------------------
def build_video_file_list(root_dir):
    """Expect: root_dir/class_name/*.mp4 (or .avi). Returns list of (video_path, class_idx) and label map."""
    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    label_map = {cls: i for i, cls in enumerate(classes)}
    files = []
    for cls in classes:
        pattern = os.path.join(root_dir, cls, "*")
        for p in glob.glob(pattern):
            if p.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                files.append((p, label_map[cls]))
    return files, label_map

def create_dataset_from_folder(root_dir, test_size=0.15, val_size=0.1, seed=42):
    files, label_map = build_video_file_list(root_dir)
    global NUM_CLASSES
    NUM_CLASSES = len(label_map)
    video_paths = [f for f, _ in files]
    labels = [l for _, l in files]
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        video_paths, labels, test_size=test_size, random_state=seed, stratify=labels)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, test_size=val_size, random_state=seed, stratify=train_labels)
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels), label_map

# --------------------------
# Classical frame feature extractor
# --------------------------
def build_frame_embedding_model(embedding_dim=EMBED_DIM, input_shape=(FRAME_SIZE[0], FRAME_SIZE[1], 3)):
    """Small CNN that maps single frame -> embedding vector."""
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(16, 3, strides=2, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(embedding_dim, activation="relu")(x)
    model = tf.keras.Model(inputs, x, name="frame_embedder")
    return model


class QuantumRNNCell(tf.keras.layers.Layer):
    """
    QRNN cell using a PennyLane QNode as the state transition.
    Input: concatenated [frame_embedding, prev_hidden] vector.
    Output: new hidden (size = n_qubits)
    """
    def __init__(self, n_qubits=N_QUBITS, n_layers=Q_CIRCUIT_LAYERS, name="quantum_rnn_cell"):
        super().__init__(name=name)
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # trainable parameters for the parametrized layers (will be shared across time steps)
        init_theta = tf.random.uniform((n_layers, n_qubits, 3), minval=-0.1, maxval=0.1)
        self.theta = tf.Variable(init_theta, dtype=tf.float32, trainable=True, name="theta")

        # classical readout weights (map measured expectations to hidden vector)
        self.readout = tf.keras.layers.Dense(self.n_qubits, activation=None, name="q_readout")

        # angle encoder (create once)
        self.angle_encoder = tf.keras.layers.Dense(self.n_qubits, activation=tf.keras.activations.tanh,
                                                   name="angle_encoder")

        # define Pennylane device and QNode
        dev = qml.device(DEVICE, wires=self.n_qubits)

        def circuit(inputs, thetas):
            # inputs: shape (n_qubits,)
            # thetas: shape (n_layers, n_qubits, 3)
            for i in range(self.n_qubits):
                qml.RY(inputs[i], wires=i)
            for l in range(self.n_layers):
                for q in range(self.n_qubits):
                    a = thetas[l, q, 0]
                    b = thetas[l, q, 1]
                    c = thetas[l, q, 2]
                    qml.RX(a, wires=q)
                    qml.RZ(b, wires=q)
                    qml.RY(c, wires=q)
                for q in range(self.n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
            return [qml.expval(qml.PauliZ(q)) for q in range(self.n_qubits)]

        # create QNode with TF interface
        self.qnode = qml.QNode(circuit, dev, interface="tf", diff_method="backprop")

        # pi constant for scaling angles
        self._pi = tf.constant(np.pi, dtype=tf.float32)

    @tf.function
    def _apply_qnode_batch(self, angles_batch, thetas):
        """
        angles_batch: (batch, n_qubits)
        thetas: tensor of shape (n_layers, n_qubits, 3)
        Returns: (batch, n_qubits) of expectation values
        """
        # tf.map_fn will call qnode(a, thetas) for each example a
        def single_call(a):
            # qnode returns shape (n_qubits,)
            return self.qnode(a, thetas)

        evs = tf.map_fn(single_call, angles_batch, fn_output_signature=tf.float32)
        return evs  # (batch, n_qubits)

    def call(self, inputs, states):
        """
        inputs: tensor shape (batch, input_dim)
        states: list with previous hidden state tensor (batch, n_qubits)
        """
        prev_h = states[0]  # (batch, n_qubits)
        # combine inputs and prev_h into a vector; reduce dimension to n_qubits via angle_encoder
        combined = tf.concat([inputs, prev_h], axis=-1)  # (batch, input_dim + n_qubits)
        angles = self.angle_encoder(combined)  # in (-1,1), shape (batch, n_qubits)
        angles = angles * self._pi  # scale to [-pi, pi]

        # ensure thetas tensor
        thetas = tf.convert_to_tensor(self.theta, dtype=tf.float32)

        # call qnode across batch using tf.map_fn (works with symbolic batch size)
        expvals = self._apply_qnode_batch(angles, thetas)  # (batch, n_qubits)

        # classical readout -> next hidden
        next_h = self.readout(expvals)  # (batch, n_qubits)
        next_h = tf.keras.activations.tanh(next_h)
        return next_h, [next_h]


class QuantumRNNLayer(tf.keras.layers.Layer):
    """
    Wraps a QuantumRNNCell to process a whole sequence.
    Uses tf.while_loop + TensorArray to be compatible with symbolic shapes.
    """
    def __init__(self, cell: QuantumRNNCell, return_sequences=False, seq_len=FRAMES_PER_CLIP, name="quantum_rnn_layer", **kwargs):
        super().__init__(name=name, **kwargs)
        self.cell = cell
        self.return_sequences = return_sequences
        # store expected sequence length (helps compute_output_shape); if user wants dynamic, pass seq_len=None
        self.seq_len = seq_len

    def call(self, inputs):
        # inputs shape: (batch, T, embed_dim)
        batch = tf.shape(inputs)[0]
        T = tf.shape(inputs)[1]
        embed_dim = tf.shape(inputs)[2]

        # initial hidden state: zeros
        h = tf.zeros((batch, self.cell.n_qubits), dtype=tf.float32)

        # TensorArray to store outputs if needed
        ta = tf.TensorArray(dtype=tf.float32, size=T)

        # loop variables: t index, h, ta
        t0 = tf.constant(0)

        def cond(t, h, ta):
            return tf.less(t, T)

        def body(t, h, ta):
            x_t = inputs[:, t, :]  # (batch, embed_dim)
            h, [h] = self.cell(x_t, [h])  # h: (batch, n_qubits)
            ta = ta.write(t, h)
            return t + 1, h, ta

        _, final_h, ta = tf.while_loop(cond, body, loop_vars=[t0, h, ta], parallel_iterations=1)

        outputs = ta.stack()  # (T, batch, n_qubits)
        outputs = tf.transpose(outputs, perm=[1, 0, 2])  # (batch, T, n_qubits)

        if self.return_sequences:
            return outputs
        else:
            return final_h  # (batch, n_qubits)

    def compute_output_shape(self, input_shape):
        # input_shape: (batch, T, embed_dim)
        batch = input_shape[0]
        T = input_shape[1] if input_shape[1] is not None else self.seq_len
        if self.return_sequences:
            return (batch, T, self.cell.n_qubits)
        else:
            return (batch, self.cell.n_qubits)


# --------------------------
# Build full model
# --------------------------
def build_qrnn_model(frame_embedder, qrnn_cell, frames_per_clip=FRAMES_PER_CLIP):
    # input: (batch, T, H, W, 3)
    video_in = tf.keras.Input(shape=(frames_per_clip, FRAME_SIZE[0], FRAME_SIZE[1], 3), name="video_input")
    # apply frame_embedder time-distributed
    td = tf.keras.layers.TimeDistributed(frame_embedder)(video_in)  # (batch, T, embed_dim)
    # quantum rnn layer
    qlayer = QuantumRNNLayer(qrnn_cell, return_sequences=False)(td)  # (batch, n_qubits)
    # optional classical post-processing
    x = tf.keras.layers.Dense(64, activation="relu")(qlayer)
    x = tf.keras.layers.Dropout(0.4)(x)
    output = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = tf.keras.Model(inputs=video_in, outputs=output, name="hybrid_qrnn")
    return model

# --------------------------
# Data generator (tf.data friendly)
# --------------------------
def gen_from_paths(paths, labels, batch_size=BATCH_SIZE, shuffle=True):
    idxs = list(range(len(paths)))
    if shuffle:
        random.shuffle(idxs)
    for start in range(0, len(paths), batch_size):
        batch_idx = idxs[start:start+batch_size]
        videos = []
        labs = []
        for i in batch_idx:
            frames = sample_frames_from_video(paths[i])
            if frames is None:
                continue
            videos.append(frames)
            labs.append(labels[i])
        if len(videos) == 0:
            continue
        videos = np.stack(videos)  # (batch, T, H, W, 3)
        labs = tf.keras.utils.to_categorical(labs, NUM_CLASSES)
        yield videos, labs

# --------------------------
# Put it all together + training loop
# --------------------------
def train_on_folder(root_dir):
    (train_p, train_l), (val_p, val_l), (test_p, test_l), label_map = create_dataset_from_folder(root_dir)
    print("Detected classes:", label_map)
    frame_embedder = build_frame_embedding_model()
    qrnn_cell = QuantumRNNCell(n_qubits=N_QUBITS, n_layers=Q_CIRCUIT_LAYERS)
    model = build_qrnn_model(frame_embedder, qrnn_cell)
    opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    # training with python generator to keep it simple (replace with tf.data pipeline if desired)
    steps_per_epoch = max(1, len(train_p) // BATCH_SIZE)
    val_steps = max(1, len(val_p) // BATCH_SIZE)
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        train_gen = gen_from_paths(train_p, train_l, batch_size=BATCH_SIZE, shuffle=True)
        pbar = tqdm(range(steps_per_epoch))
        for _ in pbar:
            try:
                x_batch, y_batch = next(train_gen)
            except StopIteration:
                break
            loss, acc = model.train_on_batch(x_batch, y_batch)
            pbar.set_description(f"loss: {loss:.4f} acc: {acc:.3f}")
        # validation
        val_gen = gen_from_paths(val_p, val_l, batch_size=BATCH_SIZE, shuffle=False)
        val_losses = []
        val_accs = []
        for _ in range(val_steps):
            try:
                xv, yv = next(val_gen)
            except StopIteration:
                break
            l, a = model.test_on_batch(xv, yv)
            val_losses.append(l)
            val_accs.append(a)
        print(f" val_loss: {np.mean(val_losses):.4f} val_acc: {np.mean(val_accs):.3f}")

    # final test evaluation
    test_gen = gen_from_paths(test_p, test_l, batch_size=BATCH_SIZE, shuffle=False)
    test_steps = max(1, len(test_p) // BATCH_SIZE)
    test_losses, test_accs = [], []
    for _ in range(test_steps):
        try:
            xt, yt = next(test_gen)
        except StopIteration:
            break
        l, a = model.test_on_batch(xt, yt)
        test_losses.append(l)
        test_accs.append(a)
    print(f"\nTest loss: {np.mean(test_losses):.4f} test_acc: {np.mean(test_accs):.3f}")
    return model, label_map

# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    DATA_ROOT = "../data/UCF-101"
    
    if not os.path.exists(DATA_ROOT):
        raise RuntimeError(f"Put your dataset in {DATA_ROOT} (class subfolders containing videos).")
    model, labels = train_on_folder(DATA_ROOT)
    # save model weights (note: QNode params are contained in tf Variables inside the model)
    model.save_weights("qrnn_weights.h5")
    print("Saved weights to qrnn_weights.h5")
