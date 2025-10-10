import { useMemo, useState } from 'react'
import UploadDropzone from '../components/UploadDropzone'
import ModelCard from '../components/ModelCard'
import MetricCard from '../components/MetricCard'
import Modal from '../components/Modal'
import Tooltip from '../components/Tooltip'
import { CLASSICAL_MODELS, QUANTUM_MODELS } from '../lib/modelInfo'
import type {
  ClassicalModelKey,
  DatasetPreview,
  QuantumModelKey,
  ComparePayload,
  CompareResult,
} from '../lib/types'
import { compareModels } from '../lib/api'
import { Play } from 'lucide-react'
import { microRoc, microPr } from '../lib/curves'
import { MetricBars, RocCurve, PrCurve } from '../components/Charts'

const HELP: Record<string, string> = {
  epochs: 'How many passes over the training data.',
  lr: 'Learning rate (how big each step is).',
  batch_size: 'How many rows are used per update.',
  C: 'Regularization strength. Smaller = stronger regularization.',
  gamma: 'RBF kernel width. “scale” is a good default.',
  n_estimators: 'Number of trees.',
  max_depth: 'Max tree depth. Leave empty for auto.',
  shots: 'Number of circuit measurements. 0 = analytic simulation.',
  noise_prob: 'Amount of depolarizing noise per layer.',
  layers: 'How many variational blocks.',
  n_qubits: 'Number of qubits in the circuit.',
}

const FRIENDLY = {
  accuracy: { name: 'Overall correctness', explain: 'How often predictions are right.' },
  f1: { name: 'Balance of precision and recall', explain: 'Helps when classes are uneven.' },
  auc: { name: 'Ranking ability', explain: 'How well positives are ranked above negatives.' },
  loss: { name: 'Penalty for wrong guesses', explain: 'Lower means fewer overconfident mistakes.' },
  latency_ms: { name: 'Run time', explain: 'How long the run took.' },
} as const

// order to display metric rows
const METRIC_ORDER: Array<keyof typeof FRIENDLY> = [
  'accuracy',
  'f1',
  'auc',
  'loss',
  'latency_ms',
]

// unified defaults
const DEFAULT_CLASSICAL: Record<ClassicalModelKey, Record<string, number | string>> = {
  mlp: { epochs: 12, lr: 0.003, batch_size: 32 },
  svm: { C: 1.0, gamma: 'scale' },
  rf: { n_estimators: 200, max_depth: '' },
  logreg: { C: 1.0 },
  mlp_torch: { epochs: 20, lr: 0.001, batch_size: 64 },
}
const DEFAULT_QUANTUM: Record<QuantumModelKey, Record<string, number | string>> = {
  qnn: { shots: 1000, noise_prob: 0.01, layers: 4, epochs: 50, lr: 0.08, n_qubits: 2 },
  vqc: { shots: 1000, noise_prob: 0.01, layers: 4, epochs: 50, lr: 0.08, n_qubits: 2 },
  qnn_simple: { epochs: 25, lr: 0.1 },
  hybrid_torch: { n_qubits: 4, layers: 2, epochs: 15, lr: 0.001, batch_size: 32 },
  aec_qnn: { encoding_dim: 4, ae_epochs: 20, batch_size: 32, q_epochs: 50, q_lr: 0.08, n_qubits: 4, layers: 4, noise_prob: 0.01, shots: 0 },
}

// one container width for everything
const CONTAINER_W = 'max-w-4xl'

export default function Compare() {
  // upload + target
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<DatasetPreview | null>(null)
  const [target, setTarget] = useState<string>('')

  // models
  const [classical, setClassical] = useState<ClassicalModelKey>('mlp')
  const [quantum, setQuantum] = useState<QuantumModelKey>('vqc')

  // params
  const [cParams, setCParams] = useState<Record<string, number | string>>(DEFAULT_CLASSICAL['mlp'])
  const [qParams, setQParams] = useState<Record<string, number | string>>(DEFAULT_QUANTUM['vqc'])

  // run + results
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<CompareResult | null>(null)

  // modal
  const [modalOpen, setModalOpen] = useState(false)
  const [modalTitle, setModalTitle] = useState('')
  const [modalBody, setModalBody] = useState('')

  function showDetails(title: string, body: string) { setModalTitle(title); setModalBody(body); setModalOpen(true) }

  function onPreview(f: File, p: DatasetPreview) {
    setFile(f); setPreview(p)
    if (!target && p.headers?.length) setTarget(p.headers[p.headers.length - 1])
  }

  function onPickClassical(key: ClassicalModelKey) { setClassical(key); setCParams(DEFAULT_CLASSICAL[key]) }
  function onPickQuantum(key: QuantumModelKey) { setQuantum(key); setQParams(DEFAULT_QUANTUM[key]) }

  async function run() {
    if (!file || !preview) return alert('Please upload a dataset first.')
    if (!target) return alert('Please choose a target column.')
    setLoading(true); setResult(null)
    try {
      const payload: ComparePayload = {
        classicalModel: classical,
        quantumModel: quantum,
        classicalParams: cParams,
        quantumParams: qParams,
        targetColumn: target,
      }
      const res = await compareModels(file, payload)
      setResult(res)
    } catch (e: any) {
      alert(e.message || 'Comparison failed')
    } finally {
      setLoading(false)
    }
  }

  // text verdict helpers
  function friendlyName(key: keyof typeof FRIENDLY) { return FRIENDLY[key].name }
  function friendlyExplain(key: keyof typeof FRIENDLY) { return FRIENDLY[key].explain }
  function explainMetric(key: keyof typeof FRIENDLY, cVal: number, qVal: number) {
    const lowerBetter = key === 'loss' || key === 'latency_ms'
    const diff = cVal - qVal, abs = Math.abs(diff)
    const near = (k: string, d: number) => k === 'latency_ms' ? d < 100 : k === 'loss' ? d < 0.02 : d < 0.01
    if (near(key, abs)) return `On ${friendlyName(key)}, both models are essentially tied.`
    const classicalWins = lowerBetter ? diff < 0 : diff > 0
    const wName = classicalWins ? CLASSICAL_MODELS[classical].name : QUANTUM_MODELS[quantum].name
    const wVal = classicalWins ? cVal : qVal
    const lVal = classicalWins ? qVal : cVal
    const sign = lowerBetter ? 'lower (better)' : 'higher (better)'
    return `${wName} is better on ${friendlyName(key)} with a ${sign} score (${wVal.toFixed(3)} vs ${lVal.toFixed(3)}).`
  }
  function overallVerdict() {
    if (!result) return ''
    const C = result.metrics.classical, Q = result.metrics.quantum
    const weights: Record<keyof typeof FRIENDLY, number> = { accuracy: 0.4, f1: 0.3, auc: 0.2, loss: 0.08, latency_ms: 0.02 }
    const keys = Object.keys(weights) as Array<keyof typeof FRIENDLY>
    let cScore = 0, qScore = 0
    for (const k of keys) {
      const c = C[k], q = Q[k]; if (c == null || q == null || Number.isNaN(c) || Number.isNaN(q)) continue
      const lower = k === 'loss' || k === 'latency_ms'
      const maxv = Math.max(c, q), minv = Math.min(c, q), eps = 1e-9
      const normC = lower ? (maxv - c) / (maxv - minv + eps) : (c - minv) / (maxv - minv + eps)
      const normQ = lower ? (maxv - q) / (maxv - minv + eps) : (q - minv) / (maxv - minv + eps)
      cScore += weights[k] * normC; qScore += weights[k] * normQ
    }
    const cname = CLASSICAL_MODELS[classical].name, qname = QUANTUM_MODELS[quantum].name
    if (Math.abs(cScore - qScore) < 0.02) return `Overall: it’s a close call. ${cname} and ${qname} perform similarly on this dataset.`
    const winner = cScore > qScore ? cname : qname
    return `Overall winner: ${winner}. This balances correctness (accuracy and F1) and ranking ability (AUC), with time considered as a minor factor.`
  }

  // charts
  const barData = useMemo(() => {
    if (!result) return []
    return [
      { metric: 'Accuracy',   Classical: result.metrics.classical.accuracy,   Quantum: result.metrics.quantum.accuracy },
      { metric: 'F1',         Classical: result.metrics.classical.f1,         Quantum: result.metrics.quantum.f1 },
      { metric: 'AUC',        Classical: result.metrics.classical.auc,        Quantum: result.metrics.quantum.auc },
      { metric: 'Loss',       Classical: result.metrics.classical.loss,       Quantum: result.metrics.quantum.loss },
      { metric: 'Time (ms)',  Classical: result.metrics.classical.latency_ms, Quantum: result.metrics.quantum.latency_ms },
    ]
  }, [result])

  const curves = useMemo(() => {
    if (!result?.diagnostics?.y_true) return null
    const y = result.diagnostics.y_true
    const pc = result.diagnostics.classical?.proba || []
    const pq = result.diagnostics.quantum?.proba || []
    return {
      rocC: microRoc(y, pc).curve,
      rocQ: microRoc(y, pq).curve,
      prC:  microPr(y, pc).curve,
      prQ:  microPr(y, pq).curve,
    }
  }, [result])

  // small input helpers
  function NumField({ label, keyName, obj, setObj }:
    { label: string; keyName: string; obj: Record<string, any>; setObj: (v: any)=>void }) {
    return (
      <div className="space-y-1">
        <div className="label">
          {label} <Tooltip text={HELP[keyName] || ''}><span className="ml-1 text-gray-400 select-none">?</span></Tooltip>
        </div>
        <input
          type="number"
          className="input w-full"
          value={String(obj[keyName] ?? '')}
          onChange={(e) => setObj({ ...obj, [keyName]: e.target.value === '' ? '' : Number(e.target.value) })}
        />
      </div>
    )
  }
  function TextField({ label, keyName, obj, setObj, placeholder }:
    { label: string; keyName: string; obj: Record<string, any>; setObj: (v: any)=>void; placeholder?: string }) {
    return (
      <div className="space-y-1">
        <div className="label">
          {label} <Tooltip text={HELP[keyName] || ''}><span className="ml-1 text-gray-400 select-none">?</span></Tooltip>
        </div>
        <input
          type="text"
          className="input w-full"
          placeholder={placeholder}
          value={String(obj[keyName] ?? '')}
          onChange={(e) => setObj({ ...obj, [keyName]: e.target.value })}
        />
      </div>
    )
  }

  function ClassicalParamCard() {
    return (
      <div className={`card mx-auto ${CONTAINER_W}`}>
        <div className="grid gap-4 md:grid-cols-3">
          {classical === 'mlp' && (<>
            <NumField label="Epochs" keyName="epochs" obj={cParams} setObj={setCParams} />
            <NumField label="LR" keyName="lr" obj={cParams} setObj={setCParams} />
            <NumField label="Batch" keyName="batch_size" obj={cParams} setObj={setCParams} />
          </>)}
          {classical === 'svm' && (<>
            <NumField label="C" keyName="C" obj={cParams} setObj={setCParams} />
            <TextField label="Gamma" keyName="gamma" obj={cParams} setObj={setCParams} placeholder="scale | auto" />
          </>)}
          {classical === 'rf' && (<>
            <NumField label="Trees" keyName="n_estimators" obj={cParams} setObj={setCParams} />
            <TextField label="Max depth" keyName="max_depth" obj={cParams} setObj={setCParams} placeholder="empty = None" />
          </>)}
          {classical === 'logreg' && (<>
            <NumField label="C" keyName="C" obj={cParams} setObj={setCParams} />
          </>)}
          {classical === 'mlp_torch' && (<>
            <NumField label="Epochs" keyName="epochs" obj={cParams} setObj={setCParams} />
            <NumField label="LR" keyName="lr" obj={cParams} setObj={setCParams} />
            <NumField label="Batch" keyName="batch_size" obj={cParams} setObj={setCParams} />
          </>)}
        </div>
      </div>
    )
  }

  function QuantumParamCard() {
    return (
      <div className={`card mx-auto ${CONTAINER_W}`}>
        <div className="grid gap-4 md:grid-cols-3">
          {(quantum === 'qnn' || quantum === 'vqc') && (<>
            <NumField label="Shots" keyName="shots" obj={qParams} setObj={setQParams} />
            <NumField label="Noise p" keyName="noise_prob" obj={qParams} setObj={setQParams} />
            <NumField label="Layers" keyName="layers" obj={qParams} setObj={setQParams} />
          </>)}
          {quantum === 'qnn_simple' && (<>
            <NumField label="Epochs" keyName="epochs" obj={qParams} setObj={setQParams} />
            <NumField label="LR" keyName="lr" obj={qParams} setObj={setQParams} />
          </>)}
          {quantum === 'hybrid_torch' && (<>
            <NumField label="Qubits" keyName="n_qubits" obj={qParams} setObj={setQParams} />
            <NumField label="Layers" keyName="layers" obj={qParams} setObj={setQParams} />
            <NumField label="Batch" keyName="batch_size" obj={qParams} setObj={setQParams} />
          </>)}
          {quantum === 'aec_qnn' && (<>
            <NumField label="Encoding dim" keyName="encoding_dim" obj={qParams} setObj={setQParams} />
            <NumField label="AE epochs" keyName="ae_epochs" obj={qParams} setObj={setQParams} />
            <NumField label="QNN epochs" keyName="q_epochs" obj={qParams} setObj={setQParams} />
          </>)}
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-slate-50">
      <div className={`mx-auto ${CONTAINER_W} px-6 py-10 space-y-10`}>
        {/* 1. Upload */}
        <section className="space-y-4">
          <h3 className="text-lg font-semibold text-slate-900 text-center">1. Upload a dataset</h3>
          <UploadDropzone onPreview={onPreview} />
          {preview && (
            <div className="card">
              <div className="grid gap-6 md:grid-cols-2">
                <div>
                  <div className="label mb-2">File</div>
                  <div className="font-semibold">{preview.filename}</div>
                  <div className="text-sm text-gray-700 mt-1">
                    Rows: {preview.nRows} · Cols: {preview.nCols} · Missing: {preview.missingCount}
                  </div>
                  <div className="mt-4">
                    <label className="label">Target column</label>
                    <select className="input mt-1 w-full" value={target} onChange={(e) => setTarget(e.target.value)}>
                      <option value="">Select...</option>
                      {preview.headers.map((h) => <option key={h} value={h}>{h}</option>)}
                    </select>
                  </div>
                </div>
                <div>
                  <div className="label mb-2">Preview</div>
                  <div className="overflow-auto max-h-64 border rounded-lg">
                    <table className="min-w-full text-sm">
                      <thead className="bg-gray-50 sticky top-0">
                        <tr>{preview.headers.map(h => <th key={h} className="px-3 py-2 text-left font-semibold">{h}</th>)}</tr>
                      </thead>
                      <tbody>
                        {preview.rows.map((r, i) => (
                          <tr key={i} className="odd:bg-white even:bg-gray-50">
                            {r.map((c, j) => <td key={j} className="px-3 py-2">{c}</td>)}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          )}
        </section>

        {/* 2. Classical */}
        <section className="space-y-4">
          <h3 className="text-lg font-semibold text-slate-900 text-center">2. Choose a classical model</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {Object.entries(CLASSICAL_MODELS).map(([key, m]) => (
              <ModelCard
                key={key}
                title={m.name}
                subtitle={m.short}
                description={m.explain}
                active={classical === key}
                onClick={() => onPickClassical(key as ClassicalModelKey)}
                onMore={() => showDetails(m.name, m.explain)}
              />
            ))}
          </div>
          <ClassicalParamCard />
        </section>

        {/* 3. Quantum */}
        <section className="space-y-4">
          <h3 className="text-lg font-semibold text-slate-900 text-center">3. Choose a quantum model</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {Object.entries(QUANTUM_MODELS).map(([key, m]) => (
              <ModelCard
                key={key}
                title={m.name}
                subtitle={m.short}
                description={m.excribe}
                active={quantum === key}
                onClick={() => onPickQuantum(key as QuantumModelKey)}
                onMore={() => showDetails(m.name, m.explain)}
              />
            ))}
          </div>
          <QuantumParamCard />
        </section>

        {/* Run button */}
        <div className="text-center">
          <button type="button" onClick={run} disabled={loading || !file} className="btn-primary inline-flex items-center gap-2">
            <Play size={16} />
            {loading ? 'Running...' : 'Run comparison'}
          </button>
        </div>

        {/* 4. Results */}
        <section className="space-y-4">
          <h3 className="text-lg font-semibold text-slate-900 text-center">4. Results</h3>
          {result ? (
            <div className="space-y-6">
              {/* Summary */}
              <div className="card text-center">
                <div className="text-sm text-gray-600">Summary</div>
                <div className="mt-1 font-semibold">
                  {CLASSICAL_MODELS[classical].name} vs {QUANTUM_MODELS[quantum].name}
                </div>
                <div className="text-sm text-gray-700">
                  Samples: {result.summary.samples}
                  {result.summary.target ? <> · Target: <span className="font-mono">{result.summary.target}</span></> : null}
                  {result.summary.n_features != null ? <> · Features: {result.summary.n_features}</> : null}
                </div>
                {result.summary.classes && result.summary.class_counts && (
                  <div className="mt-2 text-xs text-gray-700">
                    Class distribution: {result.summary.classes.map((c) => `${c}=${result.summary.class_counts?.[c] ?? 0}`).join(' · ')}
                  </div>
                )}
                {result.notes && <div className="mt-2 text-xs text-gray-600">{result.notes}</div>}
              </div>

              {/* Overall verdict */}
              <div className="card">
                <div className="font-semibold text-center">Overall verdict</div>
                <p className="text-sm text-gray-800 mt-1 text-center">{overallVerdict()}</p>
              </div>

              {/* Headline numeric cards: render as pairs so C and Q stay together */}
              <div className="space-y-3">
                {METRIC_ORDER.map((k) => (
                  <div key={k} className="grid sm:grid-cols-2 gap-3">
                    <MetricCard
                      label={`${FRIENDLY[k].name} (C)`}
                      value={result.metrics.classical[k]}
                      help={friendlyExplain(k)}
                    />
                    <MetricCard
                      label={`${FRIENDLY[k].name} (Q)`}
                      value={result.metrics.quantum[k]}
                      help={friendlyExplain(k)}
                    />
                  </div>
                ))}
              </div>

              {/* Bars */}
              <MetricBars data={barData} />

              {/* Confusion matrices */}
              {result.details && (
                <div className="grid sm:grid-cols-2 gap-3">
                  <div className="card">
                    <div className="font-semibold mb-2 text-center">Confusion (Classical)</div>
                    <ConfusionTable cm={result.details.classical.confusion} classes={result.summary.classes || []} />
                    <SmallTimings timings={result.details.classical.timings} />
                  </div>
                  <div className="card">
                    <div className="font-semibold mb-2 text-center">Confusion (Quantum)</div>
                    <ConfusionTable cm={result.details.quantum.confusion} classes={result.summary.classes || []} />
                    <SmallTimings timings={result.details.quantum.timings} />
                  </div>
                </div>
              )}

              {/* ROC & PR */}
              {curves && (
                <div className="grid sm:grid-cols-2 gap-3">
                  <RocCurve classical={curves.rocC} quantum={curves.rocQ} />
                  <PrCurve classical={curves.prC} quantum={curves.prQ} />
                </div>
              )}
            </div>
          ) : (
            <div className="text-sm text-gray-700 text-center">Run a comparison to see results.</div>
          )}
        </section>

        <Modal open={modalOpen} onClose={() => setModalOpen(false)} title={modalTitle}>
          {modalBody}
        </Modal>
      </div>
    </div>
  )
}

function ConfusionTable({ cm, classes }: { cm: number[][], classes: string[] }) {
  if (!cm?.length) return <div className="text-sm text-gray-600">No confusion matrix.</div>
  const max = Math.max(...cm.flat())
  return (
    <div className="overflow-auto">
      <table className="min-w-full text-xs border rounded">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-2 py-1">True \\ Pred →</th>
            {classes.map((c) => <th key={c} className="px-2 py-1">{c}</th>)}
          </tr>
        </thead>
        <tbody>
          {cm.map((row, i) => (
            <tr key={i} className={i % 2 ? 'bg-gray-50' : ''}>
              <td className="px-2 py-1 font-medium">{classes[i] || i}</td>
              {row.map((v, j) => {
                const a = max > 0 ? v / max : 0
                return (
                  <td key={j} className="px-2 py-1 text-right tabular-nums" style={{ backgroundColor: `rgba(59,130,246,${0.15 * a})` }}>
                    {v}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="text-xs text-gray-600 mt-2 text-center">Darker cells show where more examples landed.</div>
    </div>
  )
}

function SmallTimings({ timings }: { timings?: { train_ms: number, infer_ms: number } }) {
  if (!timings) return null
  return (
    <div className="mt-3 text-xs text-gray-600 text-center">
      Train: <span className="tabular-nums">{timings.train_ms.toFixed(1)} ms</span> ·
      Inference: <span className="tabular-nums">{timings.infer_ms.toFixed(1)} ms</span>
    </div>
  )
}
