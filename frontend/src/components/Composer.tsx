import { useRef, useState } from 'react'
import { SUGGESTED_QUESTIONS } from '../data/knowledgeBase'

interface ComposerProps {
  onSend: (text: string) => void
  onStop: () => void
  isStreaming: boolean
  showSuggestions: boolean
}

export function Composer({ onSend, onStop, isStreaming, showSuggestions }: ComposerProps) {
  const [value, setValue] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement | null>(null)

  const submit = () => {
    const text = value.trim()
    if (!text || isStreaming) return
    onSend(text)
    setValue('')
    if (textareaRef.current) textareaRef.current.style.height = 'auto'
  }

  const autoResize = () => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`
  }

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  return (
    <div className="composer-wrap">
      <div className="composer">
        {showSuggestions && (
          <div className="suggestions">
            {SUGGESTED_QUESTIONS.map((q) => (
              <button key={q} type="button" className="suggestion-chip" onClick={() => onSend(q)}>
                {q}
              </button>
            ))}
          </div>
        )}

        <div className="composer-row">
          <textarea
            ref={textareaRef}
            rows={1}
            value={value}
            placeholder="Задайте вопрос по юридической базе…"
            onChange={(e) => {
              setValue(e.target.value)
              autoResize()
            }}
            onKeyDown={onKeyDown}
            disabled={isStreaming}
          />
          {isStreaming ? (
            <button type="button" className="btn-send btn-stop" onClick={onStop} title="Остановить">
              ⏹
            </button>
          ) : (
            <button type="button" className="btn-send" onClick={submit} disabled={!value.trim()} title="Отправить">
              ➤
            </button>
          )}
        </div>

        <div className="composer-hint">Enter — отправить · Shift+Enter — новая строка</div>
      </div>
    </div>
  )
}
