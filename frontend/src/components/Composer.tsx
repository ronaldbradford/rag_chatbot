import { useEffect, useRef, useState } from 'react'

interface Props {
  disabled: boolean
  streaming: boolean
  placeholder: string
  footnote: string
  onSend: (text: string) => void
  onStop: () => void
}

export function Composer({ disabled, streaming, placeholder, footnote, onSend, onStop }: Props) {
  const [text, setText] = useState('')
  const ref = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 220)}px`
  }, [text])

  const submit = () => {
    const q = text.trim()
    if (!q || disabled || streaming) return
    onSend(q)
    setText('')
  }

  return (
    <div className="composer-wrap">
      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault()
          submit()
        }}
      >
        <textarea
          ref={ref}
          value={text}
          rows={1}
          placeholder={placeholder}
          disabled={disabled}
          aria-label="Your question"
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault()
              submit()
            }
          }}
        />
        {streaming ? (
          <button type="button" className="send is-stop" onClick={onStop}>
            Stop
          </button>
        ) : (
          <button type="submit" className="send" disabled={disabled || !text.trim()}>
            Ask
          </button>
        )}
      </form>
      <p className="composer-foot mono">{footnote}</p>
    </div>
  )
}
