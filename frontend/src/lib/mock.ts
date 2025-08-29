import { ComparePayload, CompareResult } from './types'

export async function mockCompare(_form: FormData, payload: ComparePayload): Promise<CompareResult> {
  const base = (seed: number) => (name: string) =>
    Math.round((Math.abs(Math.sin(seed + name.length)) * 0.25 + 0.7) * 1000) / 1000

  const c = base(1)
  const q = base(2)

  return new Promise((resolve) =>
    setTimeout(
      () =>
        resolve({
          summary: {
            classicalModel: payload.classicalModel,
            quantumModel: payload.quantumModel,
            samples: 1200,
          },
          metrics: {
            classical: { accuracy: c('acc'), f1: c('f1'), auc: c('auc'), loss: 0.43, latency_ms: 9 },
            quantum: { accuracy: q('acc'), f1: q('f1'), auc: q('auc'), loss: 0.51, latency_ms: 27 },
          },
          notes:
            'Mock results. When your backend is ready, set VITE_USE_MOCK=0 to call /api/compare.',
        }),
      900
    )
  )
}