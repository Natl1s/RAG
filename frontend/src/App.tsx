import { useChat } from './hooks/useChat'
import { Header } from './components/Header'
import { Chat } from './components/Chat'
import { Composer } from './components/Composer'

export default function App() {
  const { messages, isStreaming, send, stop, clear } = useChat()

  return (
    <div className="app">
      <Header onClear={clear} isStreaming={isStreaming} />
      <Chat messages={messages} />
      <Composer onSend={send} onStop={stop} isStreaming={isStreaming} />
      <footer className="app-footer">Демо-режим: ответы генерируются заглушкой.</footer>
    </div>
  )
}
