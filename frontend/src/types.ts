// Общие типы данных чат-бота.

/** События, которые отдаёт бэкенд в потоке SSE. */
export type AskEvent =
  | { type: 'meta'; conversationId: string }
  | { type: 'token'; text: string }
  | { type: 'done'; answer: string; latencyMs: number }
  | { type: 'error'; message: string }

/** Реплика диалога в том виде, в каком её хранит бэкенд. */
export interface Turn {
  role: 'user' | 'assistant'
  content: string
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: number
  latencyMs?: number
  streaming?: boolean
  pending?: boolean
}
