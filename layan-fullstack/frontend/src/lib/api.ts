// src/lib/api.ts
export type ComparePayload = {
  classicalModel: string
  quantumModel: string
  classicalParams: Record<string, any>
  quantumParams: Record<string, any>
  targetColumn: string
}

export type CompareResult = {
  summary: { classicalModel: string; quantumModel: string; samples: number }
  metrics: {
    classical: { accuracy: number; f1: number; auc: number; loss: number; latency_ms: number }
    quantum: { accuracy: number; f1: number; auc: number; loss: number; latency_ms: number }
  }
  notes?: string
}

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'

export async function compareModels(file: File, payload: ComparePayload): Promise<CompareResult> {
  const fd = new FormData()
  fd.append('file', file)
  fd.append('payload', JSON.stringify(payload))

  const r = await fetch(`${API_BASE}/api/compare`, {
    method: 'POST',
    body: fd,
  })

  if (!r.ok) {
    const text = await r.text()
    throw new Error(`HTTP ${r.status}: ${text}`)
  }
  const data = await r.json()
  if ((data as any).error) throw new Error((data as any).error)
  return data as CompareResult
}
