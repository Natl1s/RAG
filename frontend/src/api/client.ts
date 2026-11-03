// Единая точка входа в «модель» для UI.
// Режим 'mock' — браузерная заглушка, 'api' — FastAPI-заглушка (SSE).

import { askMock } from '../lib/mockEngine'
import { askApi } from './apiSse'
import type { AskEvent, EngineMode, EscalationResult, SourceChunk } from '../types'

let mode: EngineMode = 'api'

export function getEngineMode(): EngineMode {
  return mode
}

export function setEngineMode(next: EngineMode): void {
  mode = next
}

export function ask(query: string, signal?: AbortSignal): AsyncGenerator<AskEvent> {
  return mode === 'api' ? askApi(query, signal) : askMock(query, signal)
}

/**
 * Передача вопроса юристу. В демо-режиме заявка имитируется локально,
 * в режиме API — создаётся через FastAPI (`POST /api/escalate`).
 */
export async function escalate(query: string, sources: SourceChunk[] = []): Promise<EscalationResult> {
  if (mode === 'api') {
    const res = await fetch('/api/escalate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, sources }),
    })
    if (!res.ok) throw new Error(`API ответил статусом ${res.status}`)
    return (await res.json()) as EscalationResult
  }

  // Имитация сетевой задержки и создания тикета в CRM.
  await new Promise((resolve) => setTimeout(resolve, 420 + Math.random() * 380))
  return {
    ticketId: `PSB-${String(Date.now()).slice(-6)}`,
    status: 'created',
    message: 'Вопрос передан юристу ПСБ. Специалист свяжется с вами в течение 1 рабочего дня.',
  }
}
