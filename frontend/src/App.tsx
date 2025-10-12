import { Outlet, NavLink } from 'react-router-dom'
import { Cpu, Bot, Github } from 'lucide-react'

export default function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <nav className="sticky top-0 z-50 backdrop-blur-xs bg-white/70 border-b border-white/60">
        <div className="mx-auto max-w-7xl px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-9 w-9 grid place-items-center rounded-xl bg-[rgb(var(--brand))]/10 text-[rgb(var(--brand))]">
              <Bot size={20} />
            </div>
            <span className="font-extrabold tracking-tight">QML Compare</span>
          </div>

          <div className="flex items-center gap-2">
            <NavLink to="/" className={({isActive}) => `btn btn-ghost ${isActive ? 'ring-2 ring-indigo-200' : ''}`}>Home</NavLink>
            <NavLink to="/compare" className={({isActive}) => `btn btn-ghost ${isActive ? 'ring-2 ring-indigo-200' : ''}`}>
              <Cpu size={16} className="mr-2" /> Compare
            </NavLink>
            <a href="https://github.com" target="_blank" className="btn btn-ghost"><Github size={16} className="mr-2" /> GitHub</a>
          </div>
        </div>
      </nav>

      <main className="flex-1">
        <Outlet />
      </main>

      <footer className="border-t border-white/60">
        <div className="mx-auto max-w-7xl px-6 py-8 text-sm text-gray-600 text-center">
          © {new Date().getFullYear()} QML Compare • Built for Classical vs Quantum model studies
        </div>
      </footer>
    </div>
  )
}
