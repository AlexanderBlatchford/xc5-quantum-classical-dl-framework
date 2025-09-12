# learning2learn.py
# Full implementation of "Learning to Learn with Quantum Neural Networks" (PennyLane demo)

import pennylane as qml
from pennylane import qaoa
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random

# ---------------------------------------------------------------------------
# 1. Setup
# ---------------------------------------------------------------------------
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# ---------------------------------------------------------------------------
# 2. Graph Generation
# ---------------------------------------------------------------------------
def generate_graphs(n_graphs, n_nodes, p_edge):
    """Generate a list of random graphs using NetworkX."""
    datapoints = []
    for _ in range(n_graphs):
        random_graph = nx.gnp_random_graph(n_nodes, p=p_edge)
        datapoints.append(random_graph)
    return datapoints

n_graphs = 20
n_nodes = 7
p_edge = 3.0 / n_nodes
graphs = generate_graphs(n_graphs, n_nodes, p_edge)

# ---------------------------------------------------------------------------
# 3. QAOA Circuit & Cost Function
# ---------------------------------------------------------------------------
def qaoa_from_graph(graph, n_layers=1):
    wires = range(len(graph.nodes))
    cost_h, mixer_h = qaoa.maxcut(graph)

    def qaoa_layer(gamma, alpha):
        qaoa.cost_layer(gamma, cost_h)
        qaoa.mixer_layer(alpha, mixer_h)

    def circuit(params, **kwargs):
        for w in wires:
            qml.Hadamard(wires=w)
        qml.layer(qaoa_layer, n_layers, params[0], params[1])
        return qml.expval(cost_h)

    dev = qml.device("default.qubit", wires=len(graph.nodes))
    cost = qml.QNode(circuit, dev, diff_method="backprop", interface="tf")
    return cost

# Quick test
cost = qaoa_from_graph(graphs[0], n_layers=1)
x = tf.Variable([[0.5], [0.5]], dtype=tf.float32)
print("Initial cost test:", cost(x))

# ---------------------------------------------------------------------------
# 4. LSTM Cell for RNN
# ---------------------------------------------------------------------------
n_layers = 1
cell = tf.keras.layers.LSTMCell(2 * n_layers)
graph_cost_list = [qaoa_from_graph(g) for g in graphs]

# ---------------------------------------------------------------------------
# 5. RNN Iteration & Recurrent Loop
# ---------------------------------------------------------------------------
def rnn_iteration(inputs, graph_cost, n_layers=1):
    prev_cost, prev_params, prev_h, prev_c = inputs
    new_input = tf.keras.layers.concatenate([prev_cost, prev_params])
    new_params, [new_h, new_c] = cell(new_input, states=[prev_h, prev_c])
    _params = tf.reshape(new_params, shape=(2, n_layers))
    _cost = graph_cost(_params)
    new_cost = tf.reshape(tf.cast(_cost, dtype=tf.float32), shape=(1, 1))
    return [new_cost, new_params, new_h, new_c]

def recurrent_loop(graph_cost, n_layers=1, intermediate_steps=False):
    initial_cost = tf.zeros(shape=(1, 1))
    initial_params = tf.zeros(shape=(1, 2 * n_layers))
    initial_h = tf.zeros(shape=(1, 2 * n_layers))
    initial_c = tf.zeros(shape=(1, 2 * n_layers))

    out0 = rnn_iteration([initial_cost, initial_params, initial_h, initial_c], graph_cost)
    out1 = rnn_iteration(out0, graph_cost)
    out2 = rnn_iteration(out1, graph_cost)
    out3 = rnn_iteration(out2, graph_cost)
    out4 = rnn_iteration(out3, graph_cost)

    loss = tf.keras.layers.average([
        0.1 * out0[0], 0.2 * out1[0], 0.3 * out2[0], 0.4 * out3[0], 0.5 * out4[0]
    ])

    if intermediate_steps:
        return [out0[1], out1[1], out2[1], out3[1], out4[1], loss]
    else:
        return loss

# ---------------------------------------------------------------------------
# 6. Training Procedure
# ---------------------------------------------------------------------------
def train_step(graph_cost):
    with tf.GradientTape() as tape:
        loss = recurrent_loop(graph_cost)
    grads = tape.gradient(loss, cell.trainable_weights)
    opt.apply_gradients(zip(grads, cell.trainable_weights))
    return loss

opt = tf.keras.optimizers.Adam(learning_rate=0.1)
epochs = 5
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}")
    total_loss = np.array([])
    for i, graph_cost in enumerate(graph_cost_list):
        loss = train_step(graph_cost)
        total_loss = np.append(total_loss, loss.numpy())
        if i % 5 == 0:
            print(f" > Graph {i+1}/{len(graph_cost_list)} - Loss: {loss[0][0]}")
    print(f" >> Mean Loss during epoch: {np.mean(total_loss)}")

# ---------------------------------------------------------------------------
# 7. Results & Comparison with SGD
# ---------------------------------------------------------------------------
print("\n--- Testing on new graph ---")
new_graph = nx.gnp_random_graph(7, p=3 / 7)
new_cost = qaoa_from_graph(new_graph)
nx.draw(new_graph)

res = recurrent_loop(new_cost, intermediate_steps=True)
start_zeros = tf.zeros(shape=(2 * n_layers, 1))
guess_0, guess_1, guess_2, guess_3, guess_4, final_loss = res
guesses = [start_zeros, guess_0, guess_1, guess_2, guess_3, guess_4]
lstm_losses = [new_cost(tf.reshape(guess, shape=(2, n_layers))) for guess in guesses]

fig, ax = plt.subplots()
plt.plot(lstm_losses, color="blue", lw=3, ls="-.", label="LSTM")
plt.grid(ls="--", lw=2, alpha=0.25)
plt.ylabel("Cost function")
plt.xlabel("Iteration")
plt.legend()
ax.set_xticks(range(6))
plt.show()

# SGD baseline
x = tf.Variable(np.random.rand(2, 1))
opt = tf.keras.optimizers.SGD(learning_rate=0.01)
step = 15
steps, sgd_losses = [], []
for _ in range(step):
    with tf.GradientTape() as tape:
        loss = new_cost(x)
    steps.append(x)
    sgd_losses.append(loss)
    gradients = tape.gradient(loss, [x])
    opt.apply_gradients(zip(gradients, [x]))
    print(f"SGD Step {_+1} - Loss = {loss}")
print(f"Final cost (SGD): {new_cost(x).numpy()} | Optimized angles: {x.numpy()}")

fig, ax = plt.subplots()
plt.plot(sgd_losses, color="orange", lw=3, label="SGD")
plt.plot(lstm_losses, color="blue", lw=3, ls="-.", label="LSTM")
plt.grid(ls="--", lw=2, alpha=0.25)
plt.legend()
plt.ylabel("Cost function")
plt.xlabel("Iteration")
plt.show()


# Evaluate the cost function on a grid in parameter space
dx = dy = np.linspace(-1.0, 1.0, 11)
dz = np.array([new_cost([[xx], [yy]]).numpy() for yy in dy for xx in dx])
Z = dz.reshape((11, 11))

# Plot cost landscape
plt.contourf(dx, dy, Z)
plt.colorbar()

# Extract optimizer steps
params_x = [0.0] + [res[i].numpy()[0, 0] for i in range(len(res[:-1]))]
params_y = [0.0] + [res[i].numpy()[0, 1] for i in range(len(res[:-1]))]

# Plot steps
plt.plot(params_x, params_y, linestyle="--", color="red", marker="x")

plt.yticks(np.linspace(-1, 1, 5))
plt.xticks(np.linspace(-1, 1, 5))
plt.xlabel(r"$\alpha$", fontsize=12)
plt.ylabel(r"$\gamma$", fontsize=12)
plt.title("Loss Landscape", fontsize=12)
plt.show()