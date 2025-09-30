import { ClassicalModelKey, QuantumModelKey } from './types'

export const CLASSICAL_MODELS: Record<ClassicalModelKey, { name: string; short: string; explain: string }> = {
  mlp: {
    name: 'Multilayer Perceptron (MLP)',
    short: 'Dense neural network',
    explain:
      'A stack of fully-connected layers that learns non-linear patterns. Great baseline for tabular data. Key ideas: weights, activations, and backpropagation.',
  },
  cnn: {
    name: 'Convolutional Neural Network (CNN)',
    short: 'Spatial feature learner',
    explain:
      'Learns local patterns using convolutional filters (e.g., edges/textures). Ideal for images or spatial data; often more sample-efficient than plain MLPs on images.',
  },
  svm: {
    name: 'Support Vector Machine (SVM)',
    short: 'Max-margin classifier',
    explain:
      'Finds a separating boundary that maximizes the margin between classes. With kernels, it models complex decision boundaries on small/medium datasets.',
  },
}

export const QUANTUM_MODELS: Record<QuantumModelKey, { name: string; short: string; explain: string }> = {
  vqc: {
    name: 'Variational Quantum Circuit (VQC)',
    short: 'Parameterized quantum ansatz',
    explain:
      'Encodes data into qubits and optimizes a parameterized circuit to minimize a loss. Potential advantages on small/noisy data via quantum feature spaces.',
  },
  qkernel: {
    name: 'Quantum Kernel Method',
    short: 'Quantum similarity measure',
    explain:
      'Builds a kernel from state overlaps after a data-encoding circuit. A classical model (like SVM) then uses this kernel; quantum hardware evaluates similarities.',
  },
  qnn: {
    name: 'Quantum Neural Network (QNN)',
    short: 'Layered quantum blocks',
    explain:
      'Quantum analogue of neural nets using repeated encoding + trainable unitary layers. Trained with classical optimizers; supports noise-aware shots and backends.',
  },
}

export const METRIC_HELP: Record<string, string> = {
  accuracy: 'Share of correct predictions. Good first-look metric.',
  f1: 'Harmonic mean of precision & recall. Useful for class imbalance.',
  auc: 'Area under ROC curve. Measures ranking quality over thresholds.',
  loss: 'Objective minimized during training (e.g., cross-entropy).',
  latency_ms: 'End-to-end inference latency per batch in milliseconds.',
}

export const MODELS = [
  { key: "cnn", name: "Baseline CNN (MLP)", description: "Small PyTorch baseline for tabular CSV." },
  { key: "qnn", name: "Simple QNN", description: "PennyLane variational classifier." },
] as const;

export type ModelKey = typeof MODELS[number]["key"];
