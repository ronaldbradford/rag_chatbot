import { memo, useMemo } from 'react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { Source } from '../api'

interface Props {
  content: string
  sources: Source[]
  activeN: number | null
  streaming?: boolean
  onCite: (n: number) => void
}

/** Turn "[3]" and "[1, 4]" into markdown links that render as footnote marks. */
function linkCitations(text: string, valid: Set<number>) {
  return text.replace(/[ \t]*\[(\d{1,2}(?:\s*[,;]\s*\d{1,2})*)\](?!\()/g, (whole, inner: string) => {
    const nums = inner.split(/[,;]/).map((s) => Number(s.trim()))
    if (!nums.every((n) => valid.has(n))) return whole
    // A no-break space keeps the mark on the same line as the word it cites.
    return ' ' + nums.map((n) => `[${n}](#cite-${n})`).join('')
  })
}

export const Answer = memo(function Answer({ content, sources, activeN, streaming, onCite }: Props) {
  const byN = useMemo(() => new Map(sources.map((s) => [s.n, s])), [sources])
  const text = useMemo(() => linkCitations(content, new Set(byN.keys())), [content, byN])

  return (
    <div className={`prose ${streaming ? 'is-streaming' : ''}`}>
      <Markdown
        remarkPlugins={[remarkGfm]}
        components={{
          a({ href, children }) {
            const m = href?.match(/^#cite-(\d+)$/)
            if (m) {
              const src = byN.get(Number(m[1]))
              if (!src) return <>{children}</>
              return (
                <button
                  className={`cite ${activeN === src.n ? 'is-active' : ''}`}
                  onClick={() => onCite(src.n)}
                  title={`${src.document_name} · page ${src.page}`}
                  aria-label={`Source ${src.n}: ${src.document_name}, page ${src.page}`}
                >
                  {src.n}
                </button>
              )
            }
            return (
              <a href={href} target="_blank" rel="noreferrer">
                {children}
              </a>
            )
          },
        }}
      >
        {text}
      </Markdown>
    </div>
  )
})

/** Which source numbers an answer actually cites, in first-mention order. */
export function citedNumbers(content: string, sources: Source[]) {
  const valid = new Set(sources.map((s) => s.n))
  const seen: number[] = []
  for (const m of content.matchAll(/\[(\d{1,2}(?:\s*[,;]\s*\d{1,2})*)\]/g)) {
    for (const part of m[1].split(/[,;]/)) {
      const n = Number(part.trim())
      if (valid.has(n) && !seen.includes(n)) seen.push(n)
    }
  }
  return seen
}
