import { ArrowRight, Sparkles, BarChart3, Upload, Cpu, Bot, Shield, Gauge, Library, Layers, GitBranch, Zap, LineChart, Brain, Database, HelpCircle } from 'lucide-react'
import { Link } from 'react-router-dom'

export default function Landing() {
  return (
    <div>
      {/* HERO */}
      <section className="relative overflow-hidden">
        <div className="pointer-events-none -z-10 absolute inset-0 bg-gradient-to-b from-indigo-50 to-transparent" />
        <div className="mx-auto max-w-7xl px-6 pt-16 pb-10">
          <div className="grid md:grid-cols-2 gap-10 items-center">
            <div>
              <div className="inline-flex items-center gap-2 rounded-full bg-indigo-100 px-3 py-1 text-xs font-semibold text-indigo-700">
                <Sparkles size={14}/> Classical ↔ Quantum, side‑by‑side
              </div>
              <h1 className="mt-4 text-5xl font-extrabold leading-tight text-slate-800">
                Explore <span className="text-indigo-700">Classical</span> and <span className="text-indigo-700">Quantum</span> ML—without the mystery
              </h1>
              <p className="mt-4 text-lg text-slate-700">
                QML Compare is a professional sandbox for uploading datasets, trying classical and quantum models, and reading clear, human explanations for every result.
              </p>
              <div className="mt-6 flex gap-3">
                <Link to="/compare" className="btn-primary">
                  <ArrowRight size={16} className="mr-2"/>Get Started
                </Link>
                <a href="#about" className="btn-ghost">Learn more</a>
              </div>
            </div>

            <div className="card">
              <div className="text-sm text-gray-700">What you’ll see</div>
              <div className="mt-4 grid grid-cols-2 gap-4">
                <div className="card">
                  <BarChart3 className="mb-2" />
                  <div className="font-semibold text-slate-800">Clear Metrics</div>
                  <div className="text-sm text-gray-700">Accuracy, F1, AUC, loss, latency—explained.</div>
                </div>
                <div className="card">
                  <Upload className="mb-2" />
                  <div className="font-semibold text-slate-800">Easy Uploads</div>
                  <div className="text-sm text-gray-700">CSV preview + quick data checks.</div>
                </div>
                <div className="card">
                  <Cpu className="mb-2" />
                  <div className="font-semibold text-slate-800">Classical Models</div>
                  <div className="text-sm text-gray-700">MLP, CNN, SVM with sensible defaults.</div>
                </div>
                <div className="card">
                  <Bot className="mb-2" />
                  <div className="font-semibold text-slate-800">Quantum Models</div>
                  <div className="text-sm text-gray-700">VQC, Quantum Kernel, QNN—noise aware.</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* WHAT IS THIS */}
      <section id="about" className="mx-auto max-w-7xl px-6 py-10">
        <div className="grid lg:grid-cols-3 gap-6">
          <div className="card lg:col-span-2">
            <div className="subtitle">What is this?</div>
            <h2 className="mt-1 text-2xl font-bold text-slate-800">A comparison framework for frontier ML</h2>
            <p className="mt-3 text-slate-700 text-sm leading-7">
              This framework lets you run the same dataset through a classical model and a quantum model, then compare outcomes with plain‑English explanations.
              It’s built for clarity: you see metrics, latency, and trade‑offs—without digging through notebooks.
            </p>
            <div className="mt-4 grid sm:grid-cols-3 gap-3">
              <div className="card">
                <Layers className="mb-2" />
                <div className="font-semibold">Layered UI</div>
                <div className="text-sm text-gray-700">Upload → Choose → Run → Understand.</div>
              </div>
              <div className="card">
                <Gauge className="mb-2" />
                <div className="font-semibold">Explainable Metrics</div>
                <div className="text-sm text-gray-700">Hover tooltips + model notes.</div>
              </div>
              <div className="card">
                <Shield className="mb-2" />
                <div className="font-semibold">Pragmatic Defaults</div>
                <div className="text-sm text-gray-700">Sensible params; tweak when ready.</div>
              </div>
            </div>
          </div>

          <div className="card">
            <div className="subtitle">Why it exists</div>
            <h3 className="mt-1 font-semibold text-slate-800">Get signal, not hype</h3>
            <p className="mt-2 text-sm text-gray-700">
              Quantum ML is exciting, but hard to reason about. This tool demystifies it: identical data, clear metrics, concise explanations.
              If quantum helps on your dataset, you’ll see it. If not, you’ll see that too.
            </p>
          </div>
        </div>
      </section>

      {/* CLASSICAL VS QUANTUM */}
      <section className="mx-auto max-w-7xl px-6 py-6">
        <div className="grid md:grid-cols-2 gap-6">
          <div className="card">
            <div className="subtitle">Classical Models</div>
            <h3 className="mt-1 font-semibold text-slate-800">Robust, predictable, battle‑tested</h3>
            <ul className="mt-3 space-y-2 text-sm text-gray-700">
              <li className="flex gap-2"><Library size={16} className="mt-0.5" /> <span><strong>MLP:</strong> dense layers learn non‑linear patterns in tabular data.</span></li>
              <li className="flex gap-2"><Library size={16} className="mt-0.5" /> <span><strong>CNN:</strong> convolutions excel on images & spatial signals.</span></li>
              <li className="flex gap-2"><Library size={16} className="mt-0.5" /> <span><strong>SVM:</strong> max‑margin classifier; kernels shape decision boundaries.</span></li>
            </ul>
            <div className="mt-3 text-xs text-gray-600">
              Strengths: mature libraries, large‑scale performance, stable training. Trade‑offs: may need feature engineering; diminishing returns on small, noisy data.
            </div>
          </div>

          <div className="card">
            <div className="subtitle">Quantum Models</div>
            <h3 className="mt-1 font-semibold text-slate-800">Richer feature spaces, noise‑aware by design</h3>
            <ul className="mt-3 space-y-2 text-sm text-gray-700">
              <li className="flex gap-2"><Zap size={16} className="mt-0.5" /> <span><strong>VQC:</strong> parameterized circuits learned with classical optimizers.</span></li>
              <li className="flex gap-2"><Zap size={16} className="mt-0.5" /> <span><strong>Quantum Kernel:</strong> quantum‑evaluated similarities + classical SVM.</span></li>
              <li className="flex gap-2"><Zap size={16} className="mt-0.5" /> <span><strong>QNN:</strong> layered encoders + unitaries; shots & noise simulate hardware.</span></li>
            </ul>
            <div className="mt-3 text-xs text-gray-600">
              Strengths: expressive embeddings, potential for small/noisy datasets. Trade‑offs: stochastic results (shots), hardware constraints, tuning sensitivity.
            </div>
          </div>
        </div>
      </section>

      {/* USE CASES */}
      <section className="mx-auto max-w-7xl px-6 py-6">
        <div className="card">
          <div className="subtitle">What you can do</div>
          <div className="mt-3 grid md:grid-cols-3 gap-4">
            <div className="card">
              <LineChart className="mb-2" />
              <div className="font-semibold text-slate-800">Benchmark baselines</div>
              <p className="text-sm text-gray-700">Upload a CSV, compare classical vs quantum, and log outcomes you trust.</p>
            </div>
            <div className="card">
              <Brain className="mb-2" />
              <div className="font-semibold text-slate-800">Probe data regimes</div>
              <p className="text-sm text-gray-700">Vary shots/noise/epochs to see where models shine or collapse.</p>
            </div>
            <div className="card">
              <Database className="mb-2" />
              <div className="font-semibold text-slate-800">Explain results</div>
              <p className="text-sm text-gray-700">Every metric includes a short, readable explanation—no jargon dump.</p>
            </div>
          </div>
        </div>
      </section>

      {/* HOW IT WORKS */}
      <section className="mx-auto max-w-7xl px-6 py-6">
        <div className="card">
          <div className="subtitle">How it works</div>
          <ol className="mt-3 grid md:grid-cols-3 gap-4 text-sm">
            <li className="card">
              <span className="text-xs font-semibold text-gray-500">Step 1</span>
              <div className="font-semibold text-slate-800 mt-1">Upload</div>
              <p className="text-gray-700 mt-1">CSV preview, missing‑value check, choose the target column.</p>
            </li>
            <li className="card">
              <span className="text-xs font-semibold text-gray-500">Step 2</span>
              <div className="font-semibold text-slate-800 mt-1">Choose models</div>
              <p className="text-gray-700 mt-1">Pick one classical and one quantum model; tweak parameters.</p>
            </li>
            <li className="card">
              <span className="text-xs font-semibold text-gray-500">Step 3</span>
              <div className="font-semibold text-slate-800 mt-1">Compare</div>
              <p className="text-gray-700 mt-1">Run the head‑to‑head and read concise explanations for each metric.</p>
            </li>
          </ol>
        </div>
      </section>

      {/* FEATURES */}
      <section className="mx-auto max-w-7xl px-6 py-6">
        <div className="grid lg:grid-cols-4 gap-4">
          <div className="card">
            <GitBranch className="mb-2" />
            <div className="font-semibold text-slate-800">Consistent API</div>
            <p className="text-sm text-gray-700">Single contract for classical & quantum runs.</p>
          </div>
          <div className="card">
            <HelpCircle className="mb-2" />
            <div className="font-semibold text-slate-800">Built‑in explainers</div>
            <p className="text-sm text-gray-700">Tooltips & detail modals everywhere you need them.</p>
          </div>
          <div className="card">
            <Shield className="mb-2" />
            <div className="font-semibold text-slate-800">Noise & shots</div>
            <p className="text-sm text-gray-700">Realistic quantum settings, clearly surfaced.</p>
          </div>
          <div className="card">
            <Gauge className="mb-2" />
            <div className="font-semibold text-slate-800">Product‑grade UX</div>
            <p className="text-sm text-gray-700">Glass cards, thoughtful defaults, responsive layout.</p>
          </div>
        </div>
      </section>

      {/* TRUST & PRIVACY */}
      <section className="mx-auto max-w-7xl px-6 py-6">
        <div className="card flex flex-col md:flex-row items-start md:items-center gap-6">
          <Shield />
          <div>
            <div className="subtitle">Trust & privacy</div>
            <p className="text-sm text-gray-700 mt-1">
              Your uploaded CSV stays within your session for comparison. When connected to a backend, the API receives only the file and chosen parameters for evaluation.
            </p>
          </div>
        </div>
      </section>

      {/* ROADMAP */}
      <section className="mx-auto max-w-7xl px-6 py-6">
        <div className="card">
          <div className="subtitle">Roadmap</div>
          <ul className="mt-3 grid sm:grid-cols-2 lg:grid-cols-4 gap-3 text-sm text-gray-700">
            <li className="card">More classical baselines (trees, boosting)</li>
            <li className="card">Quantum backends integration options</li>
            <li className="card">Experiment tracking exports</li>
            <li className="card">Dataset health checks & profiling</li>
          </ul>
        </div>
      </section>

      {/* FAQ */}
      <section className="mx-auto max-w-7xl px-6 py-6">
        <div className="card">
          <div className="subtitle">FAQ</div>
          <div className="mt-3 grid md:grid-cols-2 gap-4">
            <details className="card">
              <summary className="font-semibold cursor-pointer">Do I need quantum hardware?</summary>
              <p className="mt-2 text-sm text-gray-700">No. You can simulate quantum behavior (shots & noise) to understand the trade‑offs before using a hardware backend.</p>
            </details>
            <details className="card">
              <summary className="font-semibold cursor-pointer">Which datasets work best?</summary>
              <p className="mt-2 text-sm text-gray-700">Start with small to medium classification datasets; quantum kernels and VQCs are often studied there.</p>
            </details>
            <details className="card">
              <summary className="font-semibold cursor-pointer">Can I export results?</summary>
              <p className="mt-2 text-sm text-gray-700">Yes—export options are on the roadmap; meanwhile you can copy metrics or screenshot cards.</p>
            </details>
            <details className="card">
              <summary className="font-semibold cursor-pointer">Is this open to new models?</summary>
              <p className="mt-2 text-sm text-gray-700">The framework is designed to add baselines. When your backend’s ready, we’ll surface them here.</p>
            </details>
          </div>
        </div>
      </section>

      {/* FINAL CTA */}
      <section className="mx-auto max-w-7xl px-6 py-10">
        <div className="card text-center">
          <h3 className="text-2xl font-bold text-slate-800">Ready to try it on your data?</h3>
          <p className="mt-2 text-sm text-gray-700">Upload a CSV and see classical vs quantum—explained clearly, side‑by‑side.</p>
          <div className="mt-5">
            <Link to="/compare" className="btn-primary">
              <ArrowRight size={16} className="mr-2"/>Get Started
            </Link>
          </div>
        </div>
      </section>
    </div>
  )
}
