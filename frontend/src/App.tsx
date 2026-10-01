import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, streamAnswer, type Conversation, type Doc, type Health, type Message } from './api'
import { Answer, citedNumbers } from './components/Answer'
import { Composer } from './components/Composer'
import { Margin } from './components/Margin'
import { Shelf } from './components/Shelf'

const SUGGESTIONS = [
  'Summarize this in five bullet points',
  'What are the main conclusions?',
  'List the key numbers, dates and names',
]

export default function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [docs, setDocs] = useState<Doc[]>([])
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [activeConv, setActiveConv] = useState<Conversation | null>(null)
  const [scope, setScope] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [focusIdx, setFocusIdx] = useState<number | null>(null)
  const [activeN, setActiveN] = useState<number | null>(null)
  const [marginOpen, setMarginOpen] = useState(false)
  const [shelfOpen, setShelfOpen] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const threadRef = useRef<HTMLDivElement>(null)
  const stickToBottom = useRef(true)

  const streaming = messages.some((m) => m.streaming)
  const currentScope = activeConv ? activeConv.document_id : scope
  const scopeDoc = docs.find((d) => d.id === currentScope)
  const readyDocs = docs.filter((d) => d.status === 'ready')

  const flash = useCallback((msg: string) => setNotice(msg), [])
  useEffect(() => {
    if (!notice) return
    const t = setTimeout(() => setNotice(null), 7000)
    return () => clearTimeout(t)
  }, [notice])

  const refreshDocs = useCallback(() => api.documents().then(setDocs).catch(() => {}), [])
  const refreshConvs = useCallback(() => api.conversations().then(setConversations).catch(() => {}), [])

  useEffect(() => {
    api.health().then(setHealth).catch(() => flash('Can’t reach the backend on port 8000. Start it and reload.'))
    refreshDocs()
    refreshConvs()
  }, [refreshDocs, refreshConvs, flash])

  // Poll while anything is still being indexed.
  const processing = docs.some((d) => d.status === 'processing')
  useEffect(() => {
    if (!processing) return
    const t = setInterval(refreshDocs, 1200)
    return () => clearInterval(t)
  }, [processing, refreshDocs])

  // ---------- uploads ----------

  const upload = useCallback(
    async (files: File[]) => {
      const pdfs = files.filter((f) => f.type === 'application/pdf' || f.name.toLowerCase().endsWith('.pdf'))
      if (pdfs.length < files.length) flash('Only PDF files can be added. Other files were skipped.')
      for (const f of pdfs) {
        try {
          const doc = await api.upload(f)
          setDocs((d) => [doc, ...d])
        } catch (e) {
          flash(`${f.name}: ${(e as Error).message}`)
        }
      }
    },
    [flash],
  )

  useEffect(() => {
    let depth = 0
    const hasFiles = (e: DragEvent) => e.dataTransfer?.types.includes('Files')
    const enter = (e: DragEvent) => {
      if (!hasFiles(e)) return
      depth++
      setDragging(true)
    }
    const leave = () => {
      depth = Math.max(0, depth - 1)
      if (depth === 0) setDragging(false)
    }
    const over = (e: DragEvent) => hasFiles(e) && e.preventDefault()
    const drop = (e: DragEvent) => {
      if (!hasFiles(e)) return
      e.preventDefault()
      depth = 0
      setDragging(false)
      upload([...(e.dataTransfer?.files ?? [])])
    }
    window.addEventListener('dragenter', enter)
    window.addEventListener('dragleave', leave)
    window.addEventListener('dragover', over)
    window.addEventListener('drop', drop)
    return () => {
      window.removeEventListener('dragenter', enter)
      window.removeEventListener('dragleave', leave)
      window.removeEventListener('dragover', over)
      window.removeEventListener('drop', drop)
    }
  }, [upload])

  // ---------- conversations ----------

  const startFresh = useCallback((docId: string | null) => {
    abortRef.current?.abort()
    setActiveConv(null)
    setMessages([])
    setScope(docId)
    setFocusIdx(null)
    setActiveN(null)
    setShelfOpen(false)
  }, [])

  const openConversation = useCallback(
    async (id: string) => {
      abortRef.current?.abort()
      try {
        const c = await api.conversation(id)
        const { messages: msgs, ...conv } = c
        setActiveConv(conv)
        setMessages(msgs)
        const last = msgs.map((m) => m.role).lastIndexOf('assistant')
        setFocusIdx(last >= 0 ? last : null)
        setActiveN(null)
        setShelfOpen(false)
        stickToBottom.current = true
        requestAnimationFrame(() => threadRef.current?.scrollTo({ top: threadRef.current.scrollHeight }))
      } catch (e) {
        flash((e as Error).message)
        refreshConvs()
      }
    },
    [flash, refreshConvs],
  )

  const deleteConversation = async (id: string) => {
    try {
      await api.deleteConversation(id)
      if (activeConv?.id === id) startFresh(activeConv.document_id)
      refreshConvs()
    } catch (e) {
      flash((e as Error).message)
    }
  }

  const deleteDoc = async (id: string) => {
    try {
      await api.deleteDocument(id)
      if (currentScope === id) startFresh(null)
      refreshDocs()
      refreshConvs()
    } catch (e) {
      flash((e as Error).message)
    }
  }

  const reindex = async (id: string) => {
    try {
      const doc = await api.reindex(id)
      setDocs((ds) => ds.map((d) => (d.id === id ? doc : d)))
    } catch (e) {
      flash((e as Error).message)
    }
  }

  // ---------- asking ----------

  const patchLast = (fn: (m: Message) => Message) =>
    setMessages((ms) => (ms.length ? [...ms.slice(0, -1), fn(ms[ms.length - 1])] : ms))

  const ask = async (question: string) => {
    let conv = activeConv
    if (!conv) {
      try {
        conv = await api.createConversation(scope)
        setActiveConv(conv)
      } catch (e) {
        flash((e as Error).message)
        return
      }
    }

    const assistantIdx = messages.length + 1
    setMessages((ms) => [
      ...ms,
      { id: `u${Date.now()}`, role: 'user', content: question, citations: [] },
      { id: `a${Date.now()}`, role: 'assistant', content: '', citations: [], streaming: true },
    ])
    setFocusIdx(assistantIdx)
    setActiveN(null)
    stickToBottom.current = true

    const ctrl = new AbortController()
    abortRef.current = ctrl
    await streamAnswer(
      conv.id,
      question,
      {
        onConversation: (c) => setActiveConv(c),
        onSources: (citations) => patchLast((m) => ({ ...m, citations })),
        onToken: (t) => patchLast((m) => ({ ...m, content: m.content + t })),
        onDone: () => patchLast((m) => ({ ...m, streaming: false })),
        onError: (error) => patchLast((m) => ({ ...m, streaming: false, error })),
      },
      ctrl.signal,
    )
    refreshConvs()
  }

  // Follow the stream unless the reader scrolled up.
  useEffect(() => {
    const el = threadRef.current
    if (el && stickToBottom.current) el.scrollTop = el.scrollHeight
  }, [messages])

  const cite = (idx: number, n: number) => {
    setFocusIdx(idx)
    setActiveN((cur) => (cur === n && focusIdx === idx ? null : n))
    setMarginOpen(true)
  }

  const focused = focusIdx !== null ? messages[focusIdx] : undefined
  const focusedSources = focused?.citations ?? []
  const focusedCited = useMemo(
    () => (focused ? citedNumbers(focused.content, focused.citations) : []),
    [focused],
  )

  // ---------- render ----------

  const scopeLabel = currentScope ? (scopeDoc?.name.replace(/\.pdf$/i, '') ?? 'Removed document') : 'All documents'
  const canAsk = health?.api_key_set !== false && (currentScope ? scopeDoc?.status === 'ready' : readyDocs.length > 0)

  const exchanges: { q?: Message; a?: Message; aIdx: number }[] = []
  messages.forEach((m, i) => {
    if (m.role === 'user') exchanges.push({ q: m, aIdx: -1 })
    else {
      const last = exchanges[exchanges.length - 1]
      if (last && !last.a) Object.assign(last, { a: m, aIdx: i })
      else exchanges.push({ a: m, aIdx: i })
    }
  })

  return (
    <div className={`app ${shelfOpen ? 'shelf-open' : ''}`}>
      <Shelf
        docs={docs}
        conversations={conversations}
        scope={currentScope}
        activeConversationId={activeConv?.id ?? null}
        onScope={startFresh}
        onUpload={upload}
        onReindex={reindex}
        onDeleteDoc={deleteDoc}
        onOpenConversation={openConversation}
        onDeleteConversation={deleteConversation}
        onNewConversation={() => startFresh(currentScope)}
      />
      <div className="scrim" onClick={() => setShelfOpen(false)} />

      <main className="desk">
        <header className="desk-head">
          <button className="icon-btn menu" aria-label="Open shelf" onClick={() => setShelfOpen(true)}>
            <svg viewBox="0 0 16 16" aria-hidden>
              <path d="M2 4h12M2 8h12M2 12h8" />
            </svg>
          </button>
          <div className="desk-title">
            <span className="eyebrow">Asking · {scopeLabel}</span>
          </div>
          {focusedSources.length > 0 && (
            <button className="text-btn margin-toggle" onClick={() => setMarginOpen(true)}>
              Sources ({focusedSources.length})
            </button>
          )}
        </header>

        {health && !health.api_key_set && (
          <div className="banner">
            Add your OpenRouter key as <code>OPENROUTER_API_KEY</code> in <code>backend/.env</code>, then restart
            the backend.
          </div>
        )}
        {notice && (
          <div className="banner is-error" role="alert">
            {notice}
            <button className="text-btn" onClick={() => setNotice(null)}>
              Dismiss
            </button>
          </div>
        )}

        <div
          className="thread"
          ref={threadRef}
          onScroll={(e) => {
            const el = e.currentTarget
            stickToBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80
          }}
        >
          {messages.length === 0 ? (
            <div className="welcome">
              {docs.length === 0 ? (
                <>
                  <h1 className="display">
                    Ask your PDFs. <em>Every answer shows its page.</em>
                  </h1>
                  <p className="lede">
                    Add a document to the shelf, ask a question in plain language, and read the exact passage each
                    answer is drawn from, right in the margin.
                  </p>
                  <label className="primary-btn">
                    Upload a PDF
                    <input
                      type="file"
                      accept="application/pdf,.pdf"
                      multiple
                      hidden
                      onChange={(e) => {
                        if (e.target.files?.length) upload([...e.target.files])
                        e.target.value = ''
                      }}
                    />
                  </label>
                </>
              ) : (
                <>
                  <h1 className="display">What would you like to know?</h1>
                  <p className="lede">
                    {currentScope
                      ? `Questions search only “${scopeLabel}”.`
                      : `Questions search all ${readyDocs.length} ready document${readyDocs.length === 1 ? '' : 's'}. Pick one on the shelf to narrow it down.`}
                  </p>
                  <div className="suggestions">
                    {SUGGESTIONS.map((s) => (
                      <button key={s} className="suggestion" disabled={!canAsk} onClick={() => ask(s)}>
                        {s}
                      </button>
                    ))}
                  </div>
                </>
              )}
            </div>
          ) : (
            exchanges.map(({ q, a, aIdx }, i) => {
              const cited = a ? citedNumbers(a.content, a.citations) : []
              const pages = [...new Set(cited.map((n) => a!.citations.find((s) => s.n === n)!.page))].sort(
                (x, y) => x - y,
              )
              return (
                <section key={(q?.id ?? a?.id) as string | number} className="exchange" style={{ ['--i' as string]: i }}>
                  {q && <h2 className="question">{q.content}</h2>}
                  {a && (
                    <div className={`answer ${aIdx === focusIdx ? 'is-focused' : ''}`}>
                      {a.streaming && !a.content ? (
                        <p className="thinking mono">
                          {a.citations.length ? `Reading ${a.citations.length} passages` : 'Searching the shelf'}
                          <span className="dots" aria-hidden />
                        </p>
                      ) : (
                        <Answer
                          content={a.content}
                          sources={a.citations}
                          streaming={a.streaming}
                          activeN={aIdx === focusIdx ? activeN : null}
                          onCite={(n) => cite(aIdx, n)}
                        />
                      )}
                      {a.error && <p className="answer-error">{a.error}</p>}
                      {!a.streaming && pages.length > 0 && (
                        <button
                          className="answer-foot mono"
                          onClick={() => {
                            setFocusIdx(aIdx)
                            setActiveN(null)
                            setMarginOpen(true)
                          }}
                        >
                          Drawn from {pages.length === 1 ? 'page' : 'pages'} {pages.join(', ')}
                        </button>
                      )}
                    </div>
                  )}
                </section>
              )
            })
          )}
        </div>

        <Composer
          disabled={!canAsk}
          streaming={streaming}
          placeholder={
            !canAsk
              ? processing
                ? 'Indexing… you can ask once it’s ready'
                : 'Add a PDF to start asking'
              : `Ask about ${currentScope ? scopeLabel : 'your documents'}…`
          }
          footnote={health ? `${health.chat_model} · answers cite their source passages` : ' '}
          onSend={ask}
          onStop={() => abortRef.current?.abort()}
        />
      </main>

      <Margin
        sources={focusedSources}
        cited={focusedCited}
        activeN={activeN}
        open={marginOpen}
        onSelect={(n) => setActiveN((cur) => (cur === n ? null : n))}
        onClose={() => setMarginOpen(false)}
      />

      {dragging && (
        <div className="dropzone" aria-hidden>
          <div className="dropzone-card">
            <span className="display-sm">Drop to add to the shelf</span>
            <span className="mono">PDF only · up to 50 MB each</span>
          </div>
        </div>
      )}
    </div>
  )
}
