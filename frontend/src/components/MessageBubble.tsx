import { useState } from 'react'
import type { Message } from '../types'

interface MessageBubbleProps {
  message: Message
}

function formatTime(ts: number): string {
  return new Date(ts).toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' })
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const [copied, setCopied] = useState(false)
  const isUser = message.role === 'user'

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(message.content)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      /* буфер обмена недоступен */
    }
  }

  return (
    <article className={`msg ${isUser ? 'msg-user' : 'msg-assistant'}`}>
      {!isUser && (
        <div className="avatar" aria-hidden="true">
          ПСБ
        </div>
      )}

      <div className="bubble">
        <div className="bubble-body">{message.content || 'Думаю…'}</div>

        <div className="bubble-meta">
          {!isUser && message.latencyMs !== undefined && !message.streaming && (
            <span className="meta-latency">{(message.latencyMs / 1000).toFixed(1)} с</span>
          )}
          <span className="meta-time">{formatTime(message.timestamp)}</span>
          {!isUser && !message.pending && (
            <button type="button" className="meta-action" onClick={copy}>
              {copied ? 'скопировано' : 'копировать'}
            </button>
          )}
        </div>
      </div>
    </article>
  )
}
