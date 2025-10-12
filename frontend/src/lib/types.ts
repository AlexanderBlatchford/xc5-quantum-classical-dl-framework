export type ClassicalModelKey = 'mlp' | 'svm' | 'rf' | 'logreg' | 'mlp_torch'
export type QuantumModelKey = 'qnn' | 'vqc' | 'qnn_simple' | 'hybrid_torch' | 'aec_qnn'

export interface ComparePayload {
  classicalModel: ClassicalModelKey
  quantumModel: QuantumModelKey
  classicalParams: Record<string, number | string>
  quantumParams: Record<string, number | string>
  targetColumn?: string
}

export interface CompareResult {
  summary: {
    classicalModel: string
    quantumModel: string
    samples: number
    target?: string
    n_features?: number
    classes?: string[]
    class_counts?: Record<string, number>
  }
  metrics: {
    classical: Record<string, number>
    quantum: Record<string, number>
  }
  details?: {
    classical: {
      confusion: number[][]
      per_class: Array<{ class: string, precision: number, recall: number, f1: number, support: number }>
      timings: { train_ms: number, infer_ms: number }
      extras?: Record<string, any>
    }
    quantum: {
      confusion: number[][]
      per_class: Array<{ class: string, precision: number, recall: number, f1: number, support: number }>
      timings: { train_ms: number, infer_ms: number }
      extras?: Record<string, any>
    }
  }
  diagnostics?: {
    y_true: number[]
    classical: { proba: number[][] }
    quantum: { proba: number[][] }
  }
  notes?: string
}
