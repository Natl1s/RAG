import { useState } from 'react'
import type { Message } from '../types'
import { Markdown } from '../lib/markdown'
import { SourcesPanel } from './SourcesPanel'

interface MessageBubbleProps {
  message: Message
  onEscalate: (message: Message) => void
}

function formatTime(ts: number): string {
  return new Date(ts).toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' })
}

export function MessageBubble({ message, onEscalate }: MessageBubbleProps) {
  const [copied, setCopied] = useState(false)
  const isUser = message.role === 'user'
  const isNotFound = message.status === 'not_found'

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
    <article className={`msg ${isUser ? 'msg-user' : 'msg-assistant'} ${isNotFound ? 'msg-notfound' : ''}`}>
      {!isUser && (
        <div className="avatar" aria-hidden="true">
          ПСБ
        </div>
      )}

      <div className="bubble">
        <div className="bubble-body">
          {message.pending && !message.content ? (
            <span className="bubble-placeholder">Ищу релевантные документы…</span>
          ) : message.streaming && !message.content ? (
            <span className="bubble-placeholder">Формирую ответ…</span>
          ) : (
            <Markdown text={message.content} />
          )}
        </div>

        {message.sources && message.sources.length > 0 && <SourcesPanel sources={message.sources} />}

        <div className="bubble-meta">
          {!isUser && message.status === 'answered' && message.confidence !== undefined && (
            <span className="chip chip-ok" title="Релевантность найденного фрагмента">
              найдено в базе · {Math.round((message.confidence ?? 0) * 100)}%
            </span>
          )}
          {!isUser && isNotFound && (
            <span className="chip chip-warn">ответа нет в базе знаний</span>
          )}
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

        {isNotFound && !message.streaming && (
          <div className="escalate">
            <button
              type="button"
              className="btn-escalate"
              onClick={() => onEscalate(message)}
              disabled={message.escalated}
            >
              {message.escalated ? 'Вопрос передан юристу' : 'Передать вопрос юристу'}
            </button>
          </div>
        )}
      </div>
    </article>
  )
}
