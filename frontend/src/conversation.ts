// Идентификатор диалога: свой у каждой вкладки, переживает F5
// и умирает вместе с вкладкой (sessionStorage чистит браузер).

const KEY = 'psb.conversationId'
const TAB_MARKER = 'psb-tab'

function newId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }
  return `c-${Date.now()}-${Math.random().toString(16).slice(2)}`
}

/**
 * id текущей вкладки.
 *
 * `window.name` переживает перезагрузку, но не копируется при дублировании
 * вкладки, поэтому по нему отсекаем унаследованный `sessionStorage`.
 */
export function getConversationId(): string {
  if (window.name !== TAB_MARKER) {
    sessionStorage.removeItem(KEY)
    window.name = TAB_MARKER
  }

  let id = sessionStorage.getItem(KEY)
  if (!id) {
    id = newId()
    sessionStorage.setItem(KEY, id)
  }
  return id
}

/** Начать новый диалог в этой же вкладке. */
export function rotateConversationId(): string {
  const id = newId()
  sessionStorage.setItem(KEY, id)
  return id
}

/**
 * Сообщить серверу о закрытии вкладки.
 *
 * `sendBeacon` умеет только POST, поэтому на бэкенде отдельный `/close`.
 * При перезагрузке beacon тоже уйдёт, но следующий запрос с тем же id
 * отменит отложенное удаление.
 */
export function closeConversation(): void {
  const id = sessionStorage.getItem(KEY)
  if (id) navigator.sendBeacon(`/api/conversations/${id}/close?ts=${Date.now()}`)
}
