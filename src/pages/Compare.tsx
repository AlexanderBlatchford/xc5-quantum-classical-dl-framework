import { useMemo, useState } from 'react'
import UploadDropzone from '../components/UploadDropzone'
import ModelCard from '../components/ModelCard'
import MetricCard from '../components/MetricCard'
import Modal from '../components/Modal'
import Tooltip from '../components/Tooltip'
import { CLASSICAL_MODELS, QUANTUM_MODELS, METRIC_HELP } from '../lib/modelInfo'
import type { ClassicalModelKey, DatasetPreview, QuantumModelKey, ComparePayload, CompareResult } from '../lib/types'
import { compareModels } from '../lib/api'
import { Play } from 'lucide-react'

const PARAM_HELP = {
  epochs: 'How many passes over the training data. More = longer training, risk of overfitting.',
  lr: 'Learning rate for the optimizer. Too high can diverge; too low can train slowly.',
  batch: 'Number of samples per gradient step. Affects stability & speed.',
  shots: 'Number of circuit measurements per evaluation. Higher = less sampling noise, more time.',
  noise: 'Probability of injected noise per gate/operation to simulate hardware imperfections.',
  layers: 'Depth of the circuit / number of variational blocks. More = expressivity, also harder to train.',
}

export default function Compare() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<DatasetPreview | null>(null)
  const [classical, setClassical] = useState<ClassicalModelKey>('mlp')
  const [quantum, setQuantum] = useState<QuantumModelKey>('vqc')
  const [target, setTarget] = useState<string>('')

  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<CompareResult | null>(null)

  // modal
  const [modalOpen, setModalOpen] = useState(false)
  const [modalTitle, setModalTitle] = useState('')
  const [modalBody, setModalBody] = useState('')

  const classicalParams = useMemo(() => ({ epochs: 12, lr: 0.003, batch_size: 32 }), [])
  const quantumParams   = useMemo(() => ({ shots: 1000, noise_prob: 0.01, layers: 4 }), [])

  function showDetails(title: string, body: string) {
    setModalTitle(title)
    setModalBody(body)
    setModalOpen(true)
  }

  function onPreview(f: File, p: DatasetPreview) {
    setFile(f); setPreview(p)
    if (!target && p.headers?.length) setTarget(p.headers[p.headers.length - 1])
  }

  async function run() {
    if (!file || !preview) return alert('Please upload a dataset first.')
    if (!target) return alert('Please choose a target column.')
    setLoading(true); setResult(null)
    try {
      const payload: ComparePayload = { classicalModel: classical, quantumModel: quantum, classicalParams, quantumParams, targetColumn: target }
      const res = await compareModels(file, payload)
      setResult(res)
    } catch (e: any) {
      alert(e.message || 'Comparison failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="mx-auto max-w-4xl px-6 py-10 space-y-8">
      <header>
        <h2 className="text-3xl font-extrabold">Compare Models</h2>
        <p className="mt-2 text-gray-700">Follow the steps below to upload data, choose models, and run a head‑to‑head comparison.</p>
      </header>

      {/* 1. Upload */}
      <section className="space-y-4">
        <h3 className="text-lg font-semibold text-slate-900">1. Upload your dataset (CSV)</h3>
        <UploadDropzone onPreview={onPreview} />
        {preview && (
          <div className="card">
            <div className="font-semibold">{preview.filename}</div>
            <div className="mt-2 grid grid-cols-2 gap-3 text-sm">
              <div><span className="text-gray-600">Rows:</span> {preview.nRows}</div>
              <div><span className="text-gray-600">Columns:</span> {preview.nCols}</div>
              <div><span className="text-gray-600">Missing:</span> {preview.missingCount}</div>
              <div>
                <label className="label">Target column</label>
                <select value={target} onChange={(e) => setTarget(e.target.value)} className="select">
                  {preview.headers.map(h => <option key={h} value={h}>{h}</option>)}
                </select>
              </div>
            </div>

            <div className="mt-4">
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
        )}
      </section>

      {/* 2. Choose Classical */}
      <section className="space-y-4">
        <h3 className="text-lg font-semibold text-slate-900">2. Choose a classical model</h3>

        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {Object.entries(CLASSICAL_MODELS).map(([key, m]) => (
            <ModelCard
              key={key}
              title={m.name}
              subtitle={m.short}
              description={m.explain}
              active={classical === key}
              onClick={() => setClassical(key as any)}
              onMore={() => showDetails(m.name, m.explain)}
              descriptionClassName="clamp-2"
            />
          ))}
        </div>

        {/* parameter row with tooltips */}
        <div className="card grid grid-cols-1 sm:grid-cols-3 gap-3">
          <div>
            <label className="label inline-flex items-center gap-2">
              Epochs
              <Tooltip text={PARAM_HELP.epochs}><span>?</span></Tooltip>
            </label>
            <input className="input" type="number" defaultValue={12} onChange={(e) => (classicalParams.epochs = +e.target.value)} />
          </div>
          <div>
            <label className="label inline-flex items-center gap-2">
              LR
              <Tooltip text={PARAM_HELP.lr}><span>?</span></Tooltip>
            </label>
            <input className="input" type="number" step="0.0001" defaultValue={0.003} onChange={(e) => (classicalParams.lr = +e.target.value)} />
          </div>
          <div>
            <label className="label inline-flex items-center gap-2">
              Batch
              <Tooltip text={PARAM_HELP.batch}><span>?</span></Tooltip>
            </label>
            <input className="input" type="number" defaultValue={32} onChange={(e) => (classicalParams.batch_size = +e.target.value)} />
          </div>
        </div>
      </section>

      {/* 3. Choose Quantum */}
      <section className="space-y-4">
        <h3 className="text-lg font-semibold text-slate-900">3. Choose a quantum model</h3>

        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {Object.entries(QUANTUM_MODELS).map(([key, m]) => (
            <ModelCard
              key={key}
              title={m.name}
              subtitle={m.short}
              description={m.explain}
              active={quantum === key}
              onClick={() => setQuantum(key as any)}
              onMore={() => showDetails(m.name, m.explain)}
              descriptionClassName="clamp-2"
            />
          ))}
        </div>

        <div className="card grid grid-cols-1 sm:grid-cols-3 gap-3">
          <div>
            <label className="label inline-flex items-center gap-2">
              Shots
              <Tooltip text={PARAM_HELP.shots}><span>?</span></Tooltip>
            </label>
            <input className="input" type="number" defaultValue={1000} onChange={(e) => (quantumParams.shots = +e.target.value)} />
          </div>
          <div>
            <label className="label inline-flex items-center gap-2">
              Noise p
              <Tooltip text={PARAM_HELP.noise}><span>?</span></Tooltip>
            </label>
            <input className="input" type="number" step="0.001" defaultValue={0.01} onChange={(e) => (quantumParams.noise_prob = +e.target.value)} />
          </div>
          <div>
            <label className="label inline-flex items-center gap-2">
              Layers
              <Tooltip text={PARAM_HELP.layers}><span>?</span></Tooltip>
            </label>
            <input className="input" type="number" defaultValue={4} onChange={(e) => (quantumParams.layers = +e.target.value)} />
          </div>
        </div>
      </section>

      {/* 4. Run (centered) */}
      <section className="flex justify-center">
        <button className="btn-primary px-5 py-2" onClick={run} disabled={loading}>
          <Play size={16} className="mr-2" />
          {loading ? 'Running…' : 'Run Comparison'}
        </button>
      </section>

      {/* 5. Results */}
      <section className="space-y-4">
        <h3 className="text-lg font-semibold text-slate-900">5. Results</h3>
        {result ? (
          <>
            <div className="card">
              <div className="text-sm text-gray-600">Summary</div>
              <div className="mt-1 font-semibold">
                {result.summary.classicalModel.toUpperCase()} vs {result.summary.quantumModel.toUpperCase()}
              </div>
              <div className="text-sm text-gray-700">Samples: {result.summary.samples}</div>
              {result.notes && <div className="mt-2 text-xs text-gray-600">{result.notes}</div>}
            </div>
            <div className="grid sm:grid-cols-2 gap-3">
              <MetricCard label="Accuracy (C)" value={result.metrics.classical.accuracy} help={METRIC_HELP.accuracy} />
              <MetricCard label="Accuracy (Q)" value={result.metrics.quantum.accuracy} help={METRIC_HELP.accuracy} />
              <MetricCard label="F1 (C)" value={result.metrics.classical.f1} help={METRIC_HELP.f1} />
              <MetricCard label="F1 (Q)" value={result.metrics.quantum.f1} help={METRIC_HELP.f1} />
              <MetricCard label="AUC (C)" value={result.metrics.classical.auc} help={METRIC_HELP.auc} />
              <MetricCard label="AUC (Q)" value={result.metrics.quantum.auc} help={METRIC_HELP.auc} />
              <MetricCard label="Loss (C)" value={result.metrics.classical.loss} help={METRIC_HELP.loss} />
              <MetricCard label="Loss (Q)" value={result.metrics.quantum.loss} help={METRIC_HELP.loss} />
              <MetricCard label="Latency ms (C)" value={result.metrics.classical.latency_ms} help={METRIC_HELP.latency_ms} />
              <MetricCard label="Latency ms (Q)" value={result.metrics.quantum.latency_ms} help={METRIC_HELP.latency_ms} />
            </div>
          </>
        ) : (
          <div className="card">
            <div className="font-semibold">Your results will appear here</div>
            <p className="text-sm text-gray-700 mt-2">
              After you run, metrics for classical and quantum models will show side‑by‑side with tooltips explaining each one.
            </p>
            <div className="mt-3 text-xs text-gray-600">
              Tip: Adjust epochs, learning rate, shots, and noise to see how results change.
            </div>
          </div>
        )}
      </section>

      {/* 6. Explanations */}
      <section className="card">
        <div className="font-semibold">What do these models do?</div>
        <p className="text-sm text-gray-700 mt-2">
          Classical models learn statistical patterns from data.
          Quantum models encode data into quantum states and measure outcomes—sometimes creating richer feature spaces.
          Results are comparable via common metrics (accuracy, F1, AUC) and a training loss.
        </p>
        <p className="text-sm text-gray-700 mt-2">
          <strong>Noise & shots:</strong> Quantum runs use finite shots (repeated measurements). Noise simulates real hardware imperfections.
        </p>
      </section>

      <Modal open={modalOpen} onClose={() => setModalOpen(false)} title={modalTitle}>
        {modalBody}
      </Modal>
    </div>
  )
}
