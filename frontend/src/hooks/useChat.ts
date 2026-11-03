import { useCallback, useEffect, useRef, useState } from 'react'
import { ask } from '../api/client'
import type { Message } from '../types'

const WELCOME: Message = {
  id: 'welcome',
  role: 'assistant',
  content:
    'Здравствуйте! Я — юридический ассистент ПСБ.\n\nЯ ищу ответы в базе юридических знаний и подсказываю, если ответа в ней нет. Задайте вопрос — например, про персональные данные, трудовой договор или налоговые вычеты.',
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
        query,
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
          if (ev.type === 'retrieval') {
            patch((m) => ({ ...m, sources: ev.sources, confidence: ev.confidence, status: ev.status, pending: false }))
          } else if (ev.type === 'token') {
            patch((m) => ({ ...m, content: m.content + ev.text }))
          } else if (ev.type === 'done') {
            finished = true
            patch((m) => ({
              ...m,
              content: ev.answer,
              sources: ev.sources,
              confidence: ev.confidence,
              status: ev.status,
              latencyMs: ev.latencyMs,
              streaming: false,
              pending: false,
            }))
          }
        }

        // Поток завершился без события done — например, пользователь нажал «Стоп».
        if (!finished) {
          patch((m) => ({
            ...m,
            content: m.content ? `${m.content}\n\n⏹ Генерация остановлена.` : '⏹ Генерация остановлена.',
            streaming: false,
            pending: false,
          }))
        }
      } catch (err) {
        const aborted = err instanceof Error && err.name === 'AbortError'
        if (aborted) {
          patch((m) => ({
            ...m,
            content: m.content ? `${m.content}\n\n⏹ Генерация остановлена.` : '⏹ Генерация остановлена.',
            streaming: false,
            pending: false,
          }))
        } else {
          const text = err instanceof Error ? err.message : 'Не удалось получить ответ'
          patch((m) => ({
            ...m,
            content: m.content ? `${m.content}\n\n⚠️ Ошибка: ${text}` : `⚠️ Ошибка: ${text}`,
            streaming: false,
            pending: false,
          }))
        }
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

  /** Добавить служебное сообщение ассистента (например, подтверждение эскалации). */
  const append = useCallback((content: string) => {
    setMessages((prev) => [...prev, { id: uid(), role: 'assistant', content, timestamp: Date.now() }])
  }, [])

  /** Отметить, что вопрос передан юристу (или снять отметку при ошибке). */
  const markEscalated = useCallback((id: string, value = true) => {
    setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, escalated: value } : m)))
  }, [])

  return { messages, isStreaming, send, stop, clear, append, markEscalated }
}
