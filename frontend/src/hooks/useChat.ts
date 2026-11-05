import { useCallback, useEffect, useRef, useState } from 'react'
import { ask } from '../api/apiSse'
import type { Message } from '../types'

const WELCOME: Message = {
  id: 'welcome',
  role: 'assistant',
  content: 'Здравствуйте! Я — юридический ассистент ПСБ. Задайте вопрос.',
  timestamp: Date.now(),
}

let counter = 0
function uid(): string {
  counter += 1
  return `m-${Date.now()}-${counter}`
}

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([WELCOME])
  const [isStreaming, setIsStreaming] = useState(false)
  const abortRef = useRef<AbortController | null>(null)

  useEffect(() => () => abortRef.current?.abort(), [])

  const send = useCallback(
    async (raw: string) => {
      const query = raw.trim()
      if (!query || isStreaming) return

      const userMsg: Message = { id: uid(), role: 'user', content: query, timestamp: Date.now() }
      const assistantId = uid()
      const assistantMsg: Message = {
        id: assistantId,
        role: 'assistant',
        content: '',
        timestamp: Date.now(),
        pending: true,
        streaming: true,
      }
      setMessages((prev) => [...prev, userMsg, assistantMsg])

      const controller = new AbortController()
      abortRef.current = controller
      setIsStreaming(true)

      const patch = (updater: (m: Message) => Message) => {
        setMessages((prev) => prev.map((m) => (m.id === assistantId ? updater(m) : m)))
      }

      let finished = false

      try {
        for await (const ev of ask(query, controller.signal)) {
          if (ev.type === 'token') {
            patch((m) => ({ ...m, content: m.content + ev.text }))
          } else if (ev.type === 'done') {
            finished = true
            patch((m) => ({
              ...m,
              content: ev.answer,
              latencyMs: ev.latencyMs,
              streaming: false,
              pending: false,
            }))
          }
        }

        // Поток завершился без события done — например, пользователь нажал «Стоп».
        if (!finished) {
          patch((m) => ({ ...m, streaming: false, pending: false }))
        }
      } catch (err) {
        const aborted = err instanceof Error && err.name === 'AbortError'
        const text = aborted ? 'Генерация остановлена.' : `Ошибка: ${err instanceof Error ? err.message : 'нет ответа'}`
        patch((m) => ({
          ...m,
          content: m.content ? `${m.content}\n\n${text}` : text,
          streaming: false,
          pending: false,
        }))
      } finally {
        setIsStreaming(false)
        abortRef.current = null
      }
    },
    [isStreaming],
  )

  const stop = useCallback(() => {
    abortRef.current?.abort()
  }, [])

  const clear = useCallback(() => {
    abortRef.current?.abort()
    setMessages([WELCOME])
  }, [])

  return { messages, isStreaming, send, stop, clear }
}
