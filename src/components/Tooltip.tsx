import React, { ReactNode, useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'

type Coords = { top: number; left: number } | null

/**
 * Renders the tooltip into document.body so it escapes parent stacking contexts,
 * backdrops, and blur effects. Positions itself below the trigger, centered.
 */
export default function Tooltip({ text, children }: { text: string; children: ReactNode }) {
  const anchorRef = useRef<HTMLSpanElement>(null)
  const [open, setOpen] = useState(false)
  const [coords, setCoords] = useState<Coords>(null)

  function place() {
    const el = anchorRef.current
    if (!el) return
    const r = el.getBoundingClientRect()
    setCoords({
      top: r.bottom + 8, // 8px gap
      left: r.left + r.width / 2,
    })
  }

  useEffect(() => {
    if (!open) return
    place()
    const onScroll = () => place()
    const onResize = () => place()
    window.addEventListener('scroll', onScroll, true)
    window.addEventListener('resize', onResize)
    return () => {
      window.removeEventListener('scroll', onScroll, true)
      window.removeEventListener('resize', onResize)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  return (
    <>
      <span
        ref={anchorRef}
        className="inline-block cursor-help text-gray-400"
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
      >
        {children}
      </span>
      {open && coords &&
        createPortal(
          <div
            style={{ position: 'fixed', top: coords.top, left: coords.left, transform: 'translateX(-50%)' }}
            className="z-[9999] rounded-md bg-gray-900 text-white text-xs px-2 py-1 shadow"
            role="tooltip"
          >
            {text}
          </div>,
          document.body
        )
      }
    </>
  )
}
