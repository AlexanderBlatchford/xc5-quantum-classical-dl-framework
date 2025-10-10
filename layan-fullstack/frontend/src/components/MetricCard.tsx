import Tooltip from './Tooltip'

export default function MetricCard({
  label,
  value,
  help,
}: {
  label: string
  value: number | string
  help?: string
}) {
  return (
    <div className="card relative overflow-visible">
      <div className="flex items-center justify-between">
        <div className="text-sm text-gray-600">{label}</div>
        {help && (
          <Tooltip text={help}>
            <span>ⓘ</span>
          </Tooltip>
        )}
      </div>
      <div className="mt-2 text-3xl font-extrabold">
        {typeof value === 'number' ? value.toFixed(3) : value}
      </div>
    </div>
  )
}
