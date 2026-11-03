import { useState } from 'react'
import type { SourceChunk } from '../types'

interface SourcesPanelProps {
  sources: SourceChunk[]
}

export function SourcesPanel({ sources }: SourcesPanelProps) {
  const [open, setOpen] = useState(false)

  return (
    <div className="sources">
      <button type="button" className="sources-toggle" onClick={() => setOpen((v) => !v)}>
        <span className={`caret ${open ? 'open' : ''}`}>▸</span>
        Источники ({sources.length})
      </button>

      {open && (
        <ul className="sources-list">
          {sources.map((s) => (
            <li key={s.id} className="source-item">
              <div className="source-head">
                <span className="source-cite">{s.cite}</span>
                <span className="source-score" title="Релевантность">
                  {(s.score * 100).toFixed(0)}%
                </span>
              </div>
              <div className="source-title">
                {s.title} · {s.section}
                {s.page ? ` · стр. ${s.page}` : ''}
              </div>
              <p className="source-text">{s.text}</p>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
