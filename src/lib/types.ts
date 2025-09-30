export type ClassicalModelKey = 'mlp' | 'cnn' | 'svm'
export type QuantumModelKey = 'vqc' | 'qkernel' | 'qnn'

export interface DatasetPreview {
  filename: string
  rows: string[][]
  headers: string[]
  nRows: number
  nCols: number
  missingCount: number
}

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
  }
  metrics: {
    classical: Record<string, number>
    quantum: Record<string, number>
  }
  notes?: string
}