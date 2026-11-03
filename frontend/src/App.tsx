import { useState } from 'react'
import { escalate, getEngineMode, setEngineMode } from './api/client'
import type { EngineMode, Message } from './types'
import { useChat } from './hooks/useChat'
import { Header } from './components/Header'
import { Chat } from './components/Chat'
import { Composer } from './components/Composer'

export default function App() {
  const { messages, isStreaming, send, stop, clear, append, markEscalated } = useChat()
  const [mode, setMode] = useState<EngineMode>(getEngineMode())

  const handleModeChange = (next: EngineMode) => {
    setEngineMode(next)
    setMode(next)
  }

  const handleEscalate = async (message: Message) => {
    if (message.escalated) return

    markEscalated(message.id, true)
    try {
      const result = await escalate(message.query ?? '', message.sources ?? [])
      append(
        `✅ ${result.message}\n\nНомер обращения: **${result.ticketId}**.\n\n` +
          '_(Это демо-эскалация: в реальной системе здесь создаётся заявка в CRM или тикет-систему.)_',
      )
    } catch (err) {
      markEscalated(message.id, false)
      const text = err instanceof Error ? err.message : 'неизвестная ошибка'
      append(`⚠️ Не удалось передать вопрос юристу: ${text}`)
    }
  }

  return (
    <div className="app">
      <Header mode={mode} onModeChange={handleModeChange} onClear={clear} isStreaming={isStreaming} />
      <Chat messages={messages} onEscalate={handleEscalate} />
      <Composer onSend={send} onStop={stop} isStreaming={isStreaming} showSuggestions={messages.length <= 1} />
      <footer className="app-footer">
        Демо-режим: ответы генерируются заглушками RAG и LLM, не являются юридической консультацией.
      </footer>
    </div>
  )
}
