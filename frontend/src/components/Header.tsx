interface HeaderProps {
  onClear: () => void
  isStreaming: boolean
}

export function Header({ onClear, isStreaming }: HeaderProps) {
  return (
    <header className="header">
      <div className="header-brand">
        <div className="brand-logo" aria-hidden="true">
          <span>ПСБ</span>
        </div>
        <div className="brand-text">
          <h1>Юридический ассистент</h1>
        </div>
      </div>

      <div className="header-controls">
        <span className={`status-pill ${isStreaming ? 'busy' : ''}`}>
          <span className="status-dot" />
          {isStreaming ? 'обработка…' : 'готов'}
        </span>

        <button type="button" className="btn-ghost" onClick={onClear} title="Очистить чат">
          Очистить
        </button>
      </div>
    </header>
  )
}
