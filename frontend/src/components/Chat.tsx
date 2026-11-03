import { useEffect, useRef } from 'react'
import type { Message } from '../types'
import { MessageBubble } from './MessageBubble'
import { TypingIndicator } from './TypingIndicator'

interface ChatProps {
  messages: Message[]
  onEscalate: (message: Message) => void
}

export function Chat({ messages, onEscalate }: ChatProps) {
  const bottomRef = useRef<HTMLDivElement | null>(null)
  const last = messages[messages.length - 1]
  const lastPending = last?.pending === true

  // Пока идёт поиск, вместо пустого пузыря показываем индикатор набора.
  const visibleMessages = messages.filter(
    (msg) => !(msg.role === 'assistant' && msg.pending && !msg.content && !msg.sources?.length),
  )

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages, lastPending])

  return (
    <main className="chat" aria-live="polite">
      <div className="chat-inner">
        {visibleMessages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} onEscalate={onEscalate} />
        ))}
        {lastPending ? <TypingIndicator /> : null}
        <div ref={bottomRef} />
      </div>
    </main>
  )
}
