import { useRef, useState } from 'react'
import { formatBytes, type Conversation, type Doc } from '../api'

interface Props {
  docs: Doc[]
  conversations: Conversation[]
  scope: string | null
  activeConversationId: string | null
  onScope: (docId: string | null) => void
  onUpload: (files: File[]) => void
  onReindex: (id: string) => void
  onDeleteDoc: (id: string) => void
  onOpenConversation: (id: string) => void
  onDeleteConversation: (id: string) => void
  onNewConversation: () => void
}

const STAGE_LABEL: Record<string, string> = {
  queued: 'Queued',
  reading: 'Reading pages',
  chunking: 'Splitting passages',
  embedding: 'Indexing',
}

function DocRow({
  doc,
  active,
  onSelect,
  onReindex,
  onDelete,
}: {
  doc: Doc
  active: boolean
  onSelect: () => void
  onReindex: () => void
  onDelete: () => void
}) {
  const [confirming, setConfirming] = useState(false)
  const usable = doc.status === 'ready'

  return (
    <li className={`doc ${active ? 'is-active' : ''} is-${doc.status}`}>
      <button className="doc-main" onClick={onSelect} disabled={!usable} title={doc.name}>
        <span className="doc-name">{doc.name.replace(/\.pdf$/i, '')}</span>
        <span className="doc-meta mono">
          {doc.status === 'processing' && `${STAGE_LABEL[doc.stage ?? 'queued'] ?? 'Working'}…`}
          {doc.status === 'ready' && `${doc.pages} pp · ${formatBytes(doc.size_bytes)}`}
          {doc.status === 'failed' && 'Couldn’t index'}
          {doc.status === 'stale' && 'Needs re-index'}
        </span>
      </button>

      {doc.status === 'processing' && (
        <div className="doc-progress" role="progressbar" aria-valuenow={Math.round(doc.progress * 100)}>
          <span style={{ transform: `scaleX(${Math.max(0.04, doc.progress)})` }} />
        </div>
      )}
      {doc.status === 'failed' && doc.error && <p className="doc-error">{doc.error}</p>}

      <div className="doc-actions">
        {(doc.status === 'failed' || doc.status === 'stale') && (
          <button className="text-btn" onClick={onReindex}>
            Re-index
          </button>
        )}
        {doc.status !== 'processing' &&
          (confirming ? (
            <>
              <button className="text-btn danger" onClick={onDelete} title="Also deletes conversations about this PDF">
                Delete + its chats
              </button>
              <button className="text-btn" onClick={() => setConfirming(false)}>
                Keep
              </button>
            </>
          ) : (
            <button className="icon-btn" aria-label={`Delete ${doc.name}`} onClick={() => setConfirming(true)}>
              <svg viewBox="0 0 16 16" aria-hidden>
                <path d="M4 4l8 8M12 4l-8 8" />
              </svg>
            </button>
          ))}
      </div>
    </li>
  )
}

export function Shelf(p: Props) {
  const input = useRef<HTMLInputElement>(null)
  const ready = p.docs.filter((d) => d.status === 'ready').length
  const docName = (id: string | null) =>
    id ? (p.docs.find((d) => d.id === id)?.name.replace(/\.pdf$/i, '') ?? 'Removed document') : 'All documents'

  return (
    <aside className="shelf">
      <header className="brand">
        <span className="lights" aria-hidden>
          <i />
          <i />
          <i />
        </span>
        <span className="wordmark">Marginalia</span>
        <span className="brand-sub mono">PDF reading room</span>
      </header>

      <section className="shelf-section">
        <div className="section-head">
          <h2 className="eyebrow">Shelf</h2>
          <button className="text-btn" onClick={() => input.current?.click()}>
            + Add PDF
          </button>
        </div>
        <input
          ref={input}
          type="file"
          accept="application/pdf,.pdf"
          multiple
          hidden
          onChange={(e) => {
            if (e.target.files?.length) p.onUpload([...e.target.files])
            e.target.value = ''
          }}
        />

        {p.docs.length === 0 ? (
          <button className="intake" onClick={() => input.current?.click()}>
            <span className="intake-title">Drop PDFs anywhere</span>
            <span className="intake-sub">or click to choose files · up to 50 MB each</span>
          </button>
        ) : (
          <ul className="doc-list">
            <li className={`doc scope-all ${p.scope === null ? 'is-active' : ''}`}>
              <button className="doc-main" onClick={() => p.onScope(null)}>
                <span className="doc-name">All documents</span>
                <span className="doc-meta mono">
                  {ready} ready · search everything
                </span>
              </button>
            </li>
            {p.docs.map((d) => (
              <DocRow
                key={d.id}
                doc={d}
                active={p.scope === d.id}
                onSelect={() => p.onScope(d.id)}
                onReindex={() => p.onReindex(d.id)}
                onDelete={() => p.onDeleteDoc(d.id)}
              />
            ))}
          </ul>
        )}
      </section>

      <section className="shelf-section grow">
        <div className="section-head">
          <h2 className="eyebrow">Conversations</h2>
          <button className="text-btn" onClick={p.onNewConversation}>
            + New
          </button>
        </div>
        {p.conversations.length === 0 ? (
          <p className="quiet">Your questions and answers are saved here.</p>
        ) : (
          <ul className="conv-list">
            {p.conversations.map((c) => (
              <li key={c.id} className={`conv ${c.id === p.activeConversationId ? 'is-active' : ''}`}>
                <button className="conv-main" onClick={() => p.onOpenConversation(c.id)}>
                  <span className="conv-title">{c.title}</span>
                  <span className="conv-meta mono">{docName(c.document_id)}</span>
                </button>
                <button
                  className="icon-btn"
                  aria-label={`Delete conversation ${c.title}`}
                  onClick={() => p.onDeleteConversation(c.id)}
                >
                  <svg viewBox="0 0 16 16" aria-hidden>
                    <path d="M4 4l8 8M12 4l-8 8" />
                  </svg>
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>
    </aside>
  )
}
