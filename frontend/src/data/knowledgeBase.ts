// Загрузка базы знаний. JSON лежит внутри frontend/knowledge и используется
// как браузерной заглушкой, так и FastAPI-бэкендом (backend/main.py).
import data from '../../knowledge/knowledge.json'
import type { KnowledgeChunk } from '../types'

interface KnowledgeData {
  chunks: KnowledgeChunk[]
  suggestedQuestions: string[]
  topics: string[]
}

const KB = data as KnowledgeData

export const KB_CHUNKS: KnowledgeChunk[] = KB.chunks
export const SUGGESTED_QUESTIONS: string[] = KB.suggestedQuestions
export const TOPICS: string[] = KB.topics
