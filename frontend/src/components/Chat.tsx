import { useEffect, useRef } from 'react'
import type { Message } from '../types'
import { MessageBubble } from './MessageBubble'
import { TypingIndicator } from './TypingIndicator'

interface ChatProps {
  messages: Message[]
}

export function Chat({ messages }: ChatProps) {
  const bottomRef = useRef<HTMLDivElement | null>(null)
  const last = messages[messages.length - 1]
  const lastPending = last?.pending === true

  // Пока ответ не начал приходить, вместо пустого пузыря показываем индикатор.
  const visible = messages.filter((m) => !(m.pending && !m.content))

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages, lastPending])

  return (
    <main className="chat" aria-live="polite">
      <div className="chat-inner">
        {visible.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}
        {lastPending ? <TypingIndicator /> : null}
        <div ref={bottomRef} />
      </div>
    </main>
  )
}
