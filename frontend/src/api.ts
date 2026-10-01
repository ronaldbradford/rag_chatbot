export type DocStatus = 'processing' | 'ready' | 'failed' | 'stale'

export interface Doc {
  id: string
  name: string
  size_bytes: number
  pages: number
  chunk_count: number
  status: DocStatus
  stage: string | null
  progress: number
  error: string | null
  created_at: string
}

export interface Source {
  n: number
  chunk_id: number
  document_id: string
  document_name: string
  page: number
  content: string
  score: number
}

export interface Conversation {
  id: string
  title: string
  document_id: string | null
  created_at: string
  updated_at: string
}

export interface Message {
  id: number | string
  role: 'user' | 'assistant'
  content: string
  citations: Source[]
  streaming?: boolean
  error?: string
}

export interface Health {
  chat_model: string
  embedding_model: string
  api_key_set: boolean
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, init)
  if (!res.ok) {
    let detail = `Request failed (${res.status})`
    try {
      const body = await res.json()
      if (typeof body.detail === 'string') detail = body.detail
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail)
  }
  return (res.status === 204 ? undefined : await res.json()) as T
}

const json = (body: unknown): RequestInit => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
})

export const api = {
  health: () => request<Health>('/health'),
  documents: () => request<Doc[]>('/documents'),
  upload(file: File) {
    const form = new FormData()
    form.append('file', file)
    return request<Doc>('/documents', { method: 'POST', body: form })
  },
  reindex: (id: string) => request<Doc>(`/documents/${id}/reindex`, { method: 'POST' }),
  deleteDocument: (id: string) => request<void>(`/documents/${id}`, { method: 'DELETE' }),
  conversations: () => request<Conversation[]>('/conversations'),
  conversation: (id: string) =>
    request<Conversation & { messages: Message[] }>(`/conversations/${id}`),
  createConversation: (document_id: string | null) =>
    request<Conversation>('/conversations', json({ document_id })),
  deleteConversation: (id: string) => request<void>(`/conversations/${id}`, { method: 'DELETE' }),
}

export interface StreamHandlers {
  onConversation?: (c: Conversation) => void
  onSources: (s: Source[]) => void
  onToken: (t: string) => void
  onDone: () => void
  onError: (message: string) => void
}

/** POST a question and read the server-sent event stream (EventSource can't POST). */
export async function streamAnswer(
  conversationId: string,
  content: string,
  h: StreamHandlers,
  signal?: AbortSignal,
) {
  let res: Response
  try {
    res = await fetch(`/api/conversations/${conversationId}/messages`, { ...json({ content }), signal })
  } catch (e) {
    if ((e as Error).name !== 'AbortError') h.onError('Could not reach the server. Is the backend running?')
    return
  }
  if (!res.ok || !res.body) {
    h.onError(`The server rejected the question (${res.status}).`)
    return
  }

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader()
  let buffer = ''
  let finished = false
  try {
    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      buffer += value
      let sep: number
      while ((sep = buffer.indexOf('\n\n')) !== -1) {
        const raw = buffer.slice(0, sep)
        buffer = buffer.slice(sep + 2)
        let event = 'message'
        let data = ''
        for (const line of raw.split('\n')) {
          if (line.startsWith('event: ')) event = line.slice(7)
          else if (line.startsWith('data: ')) data += line.slice(6)
        }
        if (!data) continue
        const payload = JSON.parse(data)
        if (event === 'conversation') h.onConversation?.(payload)
        else if (event === 'sources') h.onSources(payload)
        else if (event === 'token') h.onToken(payload)
        else if (event === 'done') {
          finished = true
          h.onDone()
        } else if (event === 'error') {
          finished = true
          h.onError(payload.message)
        }
      }
    }
  } catch (e) {
    if ((e as Error).name === 'AbortError') {
      finished = true
      h.onDone()
    } else throw e
  }
  if (!finished) h.onError('The answer stream ended unexpectedly.')
}

export function formatBytes(n: number) {
  if (n < 1024 * 1024) return `${Math.max(1, Math.round(n / 1024))} KB`
  return `${(n / 1024 / 1024).toFixed(1)} MB`
}
