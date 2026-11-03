interface TypingIndicatorProps {
  label?: string
}

export function TypingIndicator({ label = 'Поиск в базе знаний…' }: TypingIndicatorProps) {
  return (
    <div className="typing">
      <div className="typing-dots" aria-hidden="true">
        <span />
        <span />
        <span />
      </div>
      <span>{label}</span>
    </div>
  )
}
