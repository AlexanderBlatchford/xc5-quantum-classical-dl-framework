import { useRef, useState } from 'react'
import Papa from 'papaparse'
import { UploadCloud } from 'lucide-react'
import type { DatasetPreview } from '../lib/types'

export default function UploadDropzone({ onPreview }: { onPreview: (f: File, preview: DatasetPreview) => void }) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [drag, setDrag] = useState(false)

  function parse(file: File) {
    Papa.parse<string[]>(file, {
      complete: (res) => {
        const rows = res.data.filter(r => r && r.length)
        const headers = (rows[0] as string[]) || []
        const body = rows.slice(1, 21) as string[][]
        const missingCount = rows.flat().filter((c) => c === '' || c == null).length
        const preview = {
          filename: file.name,
          rows: body,
          headers,
          nRows: rows.length - 1,
          nCols: headers.length,
          missingCount,
        }
        onPreview(file, preview)
      },
      error: (err) => alert(`Parse failed: ${err.message}`),
    })
  }

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => { e.preventDefault(); setDrag(false); const f = e.dataTransfer.files?.[0]; if (f) parse(f) }}
      className={`card text-center py-10 cursor-pointer ${drag ? 'ring-2 ring-indigo-300' : ''}`}
      onClick={() => inputRef.current?.click()}
      role="button"
      aria-label="Upload dataset CSV"
      tabIndex={0}
    >
      <UploadCloud className="mx-auto mb-3" />
      <div className="font-semibold">Drop your CSV here</div>
      <div className="text-sm text-gray-600">or click to browse</div>
      <input
        ref={inputRef}
        type="file"
        accept=".csv,text/csv"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0]
          if (f) parse(f)
        }}
      />
    </div>
  )
}