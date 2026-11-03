import { useRef, useState } from 'react'

interface ComposerProps {
  onSend: (text: string) => void
  onStop: () => void
  isStreaming: boolean
}

export function Composer({ onSend, onStop, isStreaming }: ComposerProps) {
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
        <div className="composer-row">
          <textarea
            ref={textareaRef}
            rows={1}
            value={value}
            placeholder="Задайте вопрос…"
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
