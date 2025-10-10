import React from 'react'
import Tooltip from './Tooltip'

export type ParamSpec =
  | { key: string; label: string; type: 'number'; min?: number; max?: number; step?: number }
  | { key: string; label: string; type: 'text'; placeholder?: string }
  | { key: string; label: string; type: 'select'; options: string[] }

export default function ParamControls({
  title,
  specs,
  values,
  onChange,
}: {
  title: string
  specs: ParamSpec[]
  values: Record<string, number | string>
  onChange: (v: Record<string, number | string>) => void
}) {
  function set(k: string, v: string) {
    const spec = specs.find(s => s.key === k)
    let val: string | number = v
    if (spec && spec.type === 'number') {
      const n = Number(v)
      val = Number.isFinite(n) ? n : 0
    }
    onChange({ ...values, [k]: val })
  }

  return (
    <div className="card">
      <div className="font-semibold mb-3 text-center">{title}</div>
      <div className="grid gap-3 md:grid-cols-2">
        {specs.map((s) => (
          <div key={s.key}>
            <label className="label">{s.label}</label>
            {s.type === 'number' && (
              <input
                type="number"
                className="input mt-1 w-full"
                value={String(values[s.key] ?? '')}
                min={s.min}
                max={s.max}
                step={s.step}
                onChange={(e) => set(s.key, e.target.value)}
              />
            )}
            {s.type === 'text' && (
              <input
                type="text"
                className="input mt-1 w-full"
                placeholder={s.placeholder || ''}
                value={String(values[s.key] ?? '')}
                onChange={(e) => set(s.key, e.target.value)}
              />
            )}
            {s.type === 'select' && (
              <select
                className="input mt-1 w-full"
                value={String(values[s.key] ?? (s.options[0] || ''))}
                onChange={(e) => set(s.key, e.target.value)}
              >
                {s.options.map(opt => <option key={opt} value={opt}>{opt}</option>)}
              </select>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
