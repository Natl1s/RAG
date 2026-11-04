// Общие типы данных чат-бота.

/** События, которые отдаёт бэкенд в потоке SSE. */
export type AskEvent = { type: 'token'; text: string } | { type: 'done'; answer: string; latencyMs: number }

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: number
  latencyMs?: number
  streaming?: boolean
  pending?: boolean
}
