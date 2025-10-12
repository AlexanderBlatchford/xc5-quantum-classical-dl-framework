import { Info } from 'lucide-react'
import { clsx } from 'clsx'

export default function ModelCard({
  title,
  subtitle,
  description,
  active,
  onClick,
  onMore,
  className,
  descriptionClassName,
}: {
  title: string
  subtitle: string
  description: string
  active?: boolean
  onClick?: () => void
  onMore?: () => void
  className?: string
  descriptionClassName?: string
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={clsx(
        'card text-left transition hover:scale-[1.01] focus:ring-2 focus:ring-indigo-300 p-4 h-full',
        active ? 'ring-2 ring-indigo-400' : '',
        className
      )}
    >
      <div className="min-w-0">
        <div className="font-semibold">{title}</div>
        <div className="subtitle">{subtitle}</div>
        <p className={clsx('mt-2 text-sm text-gray-700', descriptionClassName)}>{description}</p>

        {/* Details link anchored at bottom-right */}
        <div className="mt-3 flex justify-end">
          <span
            onClick={(e) => { e.stopPropagation(); onMore?.() }}
            className="inline-flex items-center gap-1 text-xs font-medium text-indigo-700 hover:underline cursor-pointer"
          >
            <Info size={14} /> Details
          </span>
        </div>
      </div>
    </button>
  )
}
