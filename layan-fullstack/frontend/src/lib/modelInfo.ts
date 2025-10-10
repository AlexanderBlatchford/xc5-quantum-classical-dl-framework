import type { ClassicalModelKey, QuantumModelKey } from './types'

export const CLASSICAL_MODELS: Record<ClassicalModelKey, { name: string; short: string; explain: string }> = {
  mlp: { name: 'MLP (sklearn)', short: 'Dense NN', explain: 'Fully-connected network with ReLU, trained by Adam.' },
  svm: { name: 'SVM (RBF)', short: 'Max-margin', explain: 'Non-linear kernel SVM with probability outputs.' },
  rf:  { name: 'Random Forest', short: 'Trees ensemble', explain: 'Many trees averaged; gives feature importances.' },
  logreg: { name: 'Logistic Regression', short: 'Linear baseline', explain: 'Fast linear classifier, interpretable coefficients.' },
  mlp_torch: { name: 'MLP (PyTorch)', short: 'Dense NN (torch)', explain: 'Two-layer MLP trained with Adam (requires torch).' },
}

export const QUANTUM_MODELS: Record<QuantumModelKey, { name: string; short: string; explain: string }> = {
  qnn: { name: 'VQC OvR', short: 'Quantum variational', explain: 'Data-encoding rotations + entangling blocks; OvR heads.' },
  vqc: { name: 'VQC (alias)', short: 'Same as QNN', explain: 'Alias for the VQC OvR model.' },
  qnn_simple: { name: 'QNN (2-qubit simple)', short: 'Minimal circuit', explain: 'Simple 2-qubit circuit; uses first 2 features; OvR.' },
  hybrid_torch: { name: 'Hybrid QCNN (torch)', short: 'Torch + Pennylane', explain: 'AngleEmbedding + StronglyEntanglingLayers feeding a small head (requires torch).' },
  aec_qnn: { name: 'AEC → QNN', short: 'Autoencoder + VQC', explain: 'Keras autoencoder compresses features, then VQC trains on encoded space (requires tensorflow).' },
}

export const METRIC_HELP: Record<string, string> = {
  accuracy: 'Share of correct predictions.',
  f1: 'Macro-averaged F1 across classes.',
  auc: 'ROC AUC (OvR for multiclass).',
  loss: 'Log loss (cross-entropy).',
  latency_ms: 'End-to-end runtime for the run in milliseconds.',
}
