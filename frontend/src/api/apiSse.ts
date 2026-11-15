// Клиент FastAPI: POST /api/chat и чтение потока Server-Sent Events.
// Формат строки: `data: {json}\n\n`, завершение — `data: [DONE]`.

import type { AskEvent, Turn } from '../types'

/** Разбирает один SSE-блок и возвращает событие (или null для служебных строк). */
function parseEvent(raw: string): AskEvent | null {
  const data = raw
    .split(/\r?\n/)
    .filter((line) => line.startsWith('data:'))
    .map((line) => line.slice(5).trimStart())
    .join('\n')

  if (!data || data === '[DONE]') return null

  try {
    return JSON.parse(data) as AskEvent
  } catch {
    // Пропускаем некорректные чанки потока.
    return null
  }
}

export async function* ask(
  query: string,
  conversationId: string,
  signal?: AbortSignal,
): AsyncGenerator<AskEvent> {
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
    },
    body: JSON.stringify({ query, conversation_id: conversationId, ts: Date.now() }),
    signal,
  })

  if (!res.ok || !res.body) {
    throw new Error(`API ответил статусом ${res.status}`)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })

      // События разделяются пустой строкой; поддерживаем и \n\n, и \r\n\r\n.
      const blocks = buffer.split(/\r?\n\r?\n/)
      buffer = blocks.pop() ?? ''

      for (const block of blocks) {
        const event = parseEvent(block)
        if (event) yield event
        else if (block.includes('[DONE]')) return
      }
    }

    // Последний блок без завершающей пустой строки.
    buffer += decoder.decode()
    if (buffer.trim()) {
      const event = parseEvent(buffer)
      if (event) yield event
      else if (buffer.includes('[DONE]')) return
    }
  } finally {
    reader.releaseLock()
  }
}

/** История диалога после перезагрузки страницы. */
export async function loadConversation(id: string): Promise<Turn[]> {
  const res = await fetch(`/api/conversations/${id}?ts=${Date.now()}`)
  if (!res.ok) return []
  const data = (await res.json()) as { turns?: Turn[] }
  return data.turns ?? []
}

/** Сброс диалога по кнопке «Очистить». */
export async function resetConversation(id: string): Promise<void> {
  await fetch(`/api/conversations/${id}`, { method: 'DELETE' })
}
