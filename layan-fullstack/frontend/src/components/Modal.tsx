import { X } from 'lucide-react'
import { ReactNode, useEffect } from 'react'
import { clsx } from 'clsx'

export default function Modal({
  open,
  onClose,
  title,
  children,
  className,
}: {
  open: boolean
  onClose: () => void
  title: string
  children: ReactNode
  className?: string
}) {
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [open, onClose])

  if (!open) return null

  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center"
      aria-modal="true"
      role="dialog"
    >
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/40" onClick={onClose} />

      {/* Dialog */}
      <div className={clsx('relative z-[101] w-full max-w-2xl rounded-2xl bg-white p-6 shadow-glass', className)}>
        <div className="flex items-start justify-between gap-4">
          <h3 className="text-xl font-semibold text-slate-900">{title}</h3>
          <button
            className="rounded-lg p-2 hover:bg-gray-100"
            aria-label="Close dialog"
            onClick={onClose}
          >
            <X size={18} />
          </button>
        </div>
        <div className="mt-4 text-sm text-gray-800 leading-6">{children}</div>
      </div>
    </div>
  )
}
