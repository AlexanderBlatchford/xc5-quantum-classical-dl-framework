from planqk.quantum.sdk import PlanqkQuantumProvider
from qiskit import QuantumCircuit, transpile
import matplotlib.pyplot as plt

n_coin_tosses = 5

circuit = QuantumCircuit(n_coin_tosses)
for i in range(n_coin_tosses):
    circuit.h(i)
circuit.measure_all()

# Use the PLANQK CLI and log in with "planqk login" or set the environment variable PLANQK_PERSONAL_ACCESS_TOKEN.
# Alternatively, you can pass the access token as an argument to the constructor

provider = PlanqkQuantumProvider(access_token="plqk_ktTFQDaIrRcHFLRIuG05zQ6BjDJsNHBmA8k7F1AXm8")

# Select a quantum backend suitable for the task. All PLANQK supported quantum backends are
# listed at https://app.planqk.de/quantum-backends.
backend = provider.get_backend("azure.ionq.simulator")

# Transpile the circuit ...
circuit = transpile(circuit, backend)
# ... and run it on the backend
job = backend.run(circuit, shots=100)

counts = job.result().get_counts()

print(circuit)
print(counts)
print(type(counts))


fig, ax = plt.subplots()
ax.bar(counts.keys(), counts.values())
ax.set_xlabel("Outcome")
plt.show()
