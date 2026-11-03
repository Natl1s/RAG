import type { ReactNode } from 'react'

/**
 * Минимальный Markdown-рендер без внешних зависимостей.
 * Поддерживает: **жирный**, *курсив*, `код`, [ссылки](url),
 * заголовки (###/##), маркированные и нумерованные списки, абзацы.
 * Незакрытые конструкции (например, во время потокового вывода) остаются текстом.
 */

function renderInline(text: string, keyPrefix: string): ReactNode[] {
  const nodes: ReactNode[] = []
  const regex = /(\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\)|\*[^*]+\*)/g
  let last = 0
  let idx = 0
  let m: RegExpExecArray | null
  while ((m = regex.exec(text)) !== null) {
    if (m.index > last) nodes.push(text.slice(last, m.index))
    const tok = m[0]
    const key = `${keyPrefix}-${idx++}`
    if (tok.startsWith('**')) {
      nodes.push(<strong key={key}>{tok.slice(2, -2)}</strong>)
    } else if (tok.startsWith('`')) {
      nodes.push(<code key={key}>{tok.slice(1, -1)}</code>)
    } else if (tok.startsWith('[')) {
      const mm = tok.match(/^\[([^\]]+)\]\(([^)]+)\)$/)
      if (mm) {
        nodes.push(
          <a key={key} href={mm[2]} target="_blank" rel="noreferrer">
            {mm[1]}
          </a>,
        )
      } else {
        nodes.push(tok)
      }
    } else {
      nodes.push(<em key={key}>{tok.slice(1, -1)}</em>)
    }
    last = m.index + tok.length
  }
  if (last < text.length) nodes.push(text.slice(last))
  return nodes
}

function parseBlocks(text: string): ReactNode[] {
  const lines = text.split('\n')
  const out: ReactNode[] = []
  let key = 0
  let listKind: 'ul' | 'ol' | null = null
  let listItems: string[] = []

  const flushList = () => {
    if (listItems.length === 0) return
    const items = listItems.map((it, j) => (
      <li key={`li-${key}-${j}`}>{renderInline(it, `li-${key}-${j}`)}</li>
    ))
    out.push(listKind === 'ol' ? <ol key={`ol-${key}`}>{items}</ol> : <ul key={`ul-${key}`}>{items}</ul>)
    listItems = []
    listKind = null
    key += 1
  }

  for (const raw of lines) {
    const line = raw.trimEnd()
    if (line.trim() === '') {
      flushList()
      continue
    }

    const heading = line.match(/^(#{1,3})\s+(.*)$/)
    if (heading) {
      flushList()
      const level = heading[1].length
      const content = heading[2]
      if (level === 1) out.push(<h3 key={`h-${key++}`}>{renderInline(content, `h-${key}`)}</h3>)
      else if (level === 2) out.push(<h4 key={`h-${key++}`}>{renderInline(content, `h-${key}`)}</h4>)
      else out.push(<h5 key={`h-${key++}`}>{renderInline(content, `h-${key}`)}</h5>)
      continue
    }

    const ulMatch = line.match(/^\s*[-•*]\s+(.*)$/)
    if (ulMatch) {
      if (listKind !== 'ul') {
        flushList()
        listKind = 'ul'
      }
      listItems.push(ulMatch[1])
      continue
    }

    const olMatch = line.match(/^\s*(\d+)[.)]\s+(.*)$/)
    if (olMatch) {
      if (listKind !== 'ol') {
        flushList()
        listKind = 'ol'
      }
      listItems.push(olMatch[2])
      continue
    }

    flushList()
    out.push(<p key={`p-${key++}`}>{renderInline(line, `p-${key}`)}</p>)
  }
  flushList()
  return out
}

export function Markdown({ text }: { text: string }): ReactNode {
  return <div className="md">{parseBlocks(text)}</div>
}
