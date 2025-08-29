import { ComparePayload, CompareResult } from './types'
import { mockCompare } from './mock'

const USE_MOCK = import.meta.env.VITE_USE_MOCK === '1'

export async function compareModels(file: File, payload: ComparePayload): Promise<CompareResult> {
  const form = new FormData()
  form.append('file', file)
  form.append('payload', JSON.stringify(payload))

  if (USE_MOCK) {
    return mockCompare(form, payload)
  }

  const res = await fetch('/api/compare', { method: 'POST', body: form })
  if (!res.ok) throw new Error(`Compare failed: ${res.status} ${res.statusText}`)
  return (await res.json()) as CompareResult
}