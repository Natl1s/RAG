import type { EngineMode } from '../types'

interface HeaderProps {
  mode: EngineMode
  onModeChange: (mode: EngineMode) => void
  onClear: () => void
  isStreaming: boolean
}

export function Header({ mode, onModeChange, onClear, isStreaming }: HeaderProps) {
  return (
    <header className="header">
      <div className="header-brand">
        <div className="brand-logo" aria-hidden="true">
          <span>ПСБ</span>
        </div>
        <div className="brand-text">
          <h1>Юридический ассистент</h1>
          <p>Поиск по базе юридических знаний · RAG + LLM</p>
        </div>
      </div>

      <div className="header-controls">
        <div className="mode-switch" role="group" aria-label="Режим модели">
          <button
            type="button"
            className={mode === 'mock' ? 'active' : ''}
            onClick={() => onModeChange('mock')}
            title="Браузерная заглушка моделей"
            disabled={isStreaming}
          >
            Заглушка
          </button>
          <button
            type="button"
            className={mode === 'api' ? 'active' : ''}
            onClick={() => onModeChange('api')}
            title="FastAPI-заглушка (требуется запущенный backend)"
            disabled={isStreaming}
          >
            API
          </button>
        </div>

        <span className={`status-pill ${isStreaming ? 'busy' : ''}`} title="Статус конвейера">
          <span className="status-dot" />
          {isStreaming ? 'обработка…' : mode === 'api' ? 'API · FastAPI' : 'демо · заглушки моделей'}
        </span>

        <button type="button" className="btn-ghost" onClick={onClear} title="Очистить чат">
          Очистить
        </button>
      </div>
    </header>
  )
}
