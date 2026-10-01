import { useEffect, useRef } from 'react'
import type { Source } from '../api'

interface Props {
  sources: Source[]
  cited: number[]
  activeN: number | null
  open: boolean
  onSelect: (n: number) => void
  onClose: () => void
}

/** Re-flow PDF text into paragraphs, dropping hard line wraps. */
const paragraphs = (text: string) =>
  text
    .split(/\n\s*\n/)
    .map((para) => para.replace(/\s*\n\s*/g, ' ').trim())
    .filter(Boolean)

function Note({ src, active, onSelect }: { src: Source; active: boolean; onSelect: () => void }) {
  const ref = useRef<HTMLElement>(null)

  useEffect(() => {
    if (active) ref.current?.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
  }, [active])

  return (
    <article ref={ref} className={`note ${active ? 'is-active' : ''}`}>
      <button className="note-head" onClick={onSelect} aria-expanded={active}>
        <span className="note-n mono">{src.n}</span>
        <span className="note-where">
          <span className="note-doc">{src.document_name.replace(/\.pdf$/i, '')}</span>
          <span className="note-page mono">page {src.page}</span>
        </span>
      </button>
      <blockquote className="note-body">
        {paragraphs(src.content).map((para, i) => (
          <p key={i}>
            {/* key forces the highlighter stroke to replay when re-selected */}
            <mark key={active ? 'on' : 'off'} className={active ? 'swipe' : ''}>
              {para}
            </mark>
          </p>
        ))}
      </blockquote>
      <a
        className="note-open mono"
        href={`/api/documents/${src.document_id}/file#page=${src.page}`}
        target="_blank"
        rel="noreferrer"
      >
        Open PDF at page {src.page} ↗
      </a>
    </article>
  )
}

export function Margin({ sources, cited, activeN, open, onSelect, onClose }: Props) {
  const byN = new Map(sources.map((s) => [s.n, s]))
  const citedSources = cited.map((n) => byN.get(n)!).filter(Boolean)
  const others = sources.filter((s) => !cited.includes(s.n))

  return (
    <aside className={`margin ${open ? 'is-open' : ''}`} aria-label="Sources">
      <div className="section-head margin-head">
        <h2 className="eyebrow">In the margin</h2>
        <button className="text-btn margin-close" onClick={onClose}>
          Close
        </button>
      </div>

      {sources.length === 0 ? (
        <p className="quiet margin-empty">
          Passages behind each answer appear here. Select a footnote number in an answer to read the
          exact text it came from.
        </p>
      ) : (
        <div className="notes">
          {citedSources.map((s) => (
            <Note key={s.n} src={s} active={activeN === s.n} onSelect={() => onSelect(s.n)} />
          ))}
          {others.length > 0 && (
            <>
              <h3 className="eyebrow notes-sub">Also consulted</h3>
              {others.map((s) => (
                <Note key={s.n} src={s} active={activeN === s.n} onSelect={() => onSelect(s.n)} />
              ))}
            </>
          )}
        </div>
      )}
    </aside>
  )
}
