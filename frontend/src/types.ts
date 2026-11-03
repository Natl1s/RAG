// Общие типы данных для чат-бота юридического ассистента.

/** Фрагмент (чанк) базы знаний — «сырой» документ без вычисленной релевантности. */
export interface KnowledgeChunk {
  id: string
  docId: string
  title: string
  /** Короткая ссылка на норму, например «152-ФЗ, ст. 9». */
  cite: string
  section: string
  page: number
  keywords: string[]
  /** Текст, который «извлекает» RAG-пайплайн (показывается как источник). */
  text: string
  /** Готовый ответ, который «генерирует» LLM-заглушка. */
  answer: string
  /** Краткий вывод для комбинированных ответов. */
  summary: string
}

/** Источник, возвращаемый ретривером: чанк + вычисленная релевантность. */
export interface SourceChunk extends KnowledgeChunk {
  /** Нормализованная релевантность 0..1. */
  score: number
}

export type AnswerStatus = 'answered' | 'not_found'

/**
 * События, которые выдаёт конвейер «RAG + LLM».
 * Это единый интерфейс и для браузерной заглушки, и для FastAPI (SSE).
 */
export type AskEvent =
  | { type: 'retrieval'; sources: SourceChunk[]; confidence: number; status: AnswerStatus }
  | { type: 'token'; text: string }
  | { type: 'done'; answer: string; sources: SourceChunk[]; confidence: number; status: AnswerStatus; latencyMs: number }

/** Режим движка: браузерная заглушка или реальный FastAPI. */
export type EngineMode = 'mock' | 'api'

/** Результат передачи вопроса юристу (эскалации). */
export interface EscalationResult {
  ticketId: string
  status: 'created'
  message: string
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: number
  status?: AnswerStatus
  sources?: SourceChunk[]
  confidence?: number
  latencyMs?: number
  streaming?: boolean
  pending?: boolean
  /** Вопрос пользователя, на который отвечает это сообщение ассистента. */
  query?: string
  /** Вопрос уже передан юристу. */
  escalated?: boolean
}
