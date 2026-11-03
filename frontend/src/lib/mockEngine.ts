// Браузерная заглушка конвейера «RAG + LLM».
// Здесь воспроизводится та же логика, что и в backend/main.py (FastAPI-заглушка),
// чтобы фронтенд работал полностью автономно без сервера.

import { KB_CHUNKS, TOPICS } from '../data/knowledgeBase'
import type { AskEvent, SourceChunk } from '../types'

const STOP_WORDS = new Set([
  'как', 'что', 'какой', 'какая', 'какое', 'какие', 'каков', 'какова', 'каковы',
  'кто', 'где', 'когда', 'почему', 'зачем', 'отчего', 'сколько', 'чей', 'чья', 'чьё', 'чьи',
  'нужно', 'надо', 'необходимо', 'можно', 'могу', 'можешь', 'может', 'можем', 'могут', 'следует', 'стоит',
  'для', 'при', 'под', 'над', 'перед', 'после', 'между', 'через', 'вокруг', 'около', 'без', 'вне', 'внутри',
  'это', 'этот', 'эта', 'эти', 'этого', 'этому', 'этим', 'этом', 'тот', 'та', 'то', 'те', 'такой', 'такая', 'такое', 'такие',
  'есть', 'если', 'или', 'либо', 'и', 'а', 'но', 'да', 'же', 'ли', 'бы', 'то', 'так', 'вот', 'вон', 'уже', 'ещё', 'тоже', 'также', 'только', 'просто', 'вообще',
  'в', 'во', 'на', 'с', 'со', 'от', 'ото', 'до', 'по', 'за', 'из', 'изо', 'о', 'об', 'обо', 'у', 'к', 'ко', 'не', 'ни', 'нет',
  'мне', 'меня', 'мой', 'моя', 'моё', 'мои', 'моего', 'моей', 'моим', 'моем', 'вы', 'вам', 'вас', 'ваш', 'ваша', 'ваше', 'ваши',
  'я', 'он', 'она', 'оно', 'они', 'мы', 'ты', 'себе', 'себя', 'свой', 'своя', 'своё', 'свои',
  'весь', 'вся', 'всё', 'все', 'всего', 'всей', 'всем', 'буду', 'будет', 'будут', 'быть', 'был', 'была', 'были', 'было',
  'данных', 'данные', 'данным', 'данными', 'данного', 'данной',
  // Обобщённые единицы времени и денег — не должны вызывать кросс-совпадения.
  'дней', 'день', 'дня', 'дни', 'лет', 'год', 'года', 'годов', 'месяц', 'месяца', 'месяцев',
  'неделя', 'недели', 'недель', 'тысяч', 'тысячи', 'тысячу', 'млн', 'тыс', 'руб', 'рубля', 'рублей',
])

const RETRIEVAL_MIN_SCORE = 2
const ANSWER_MIN_SCORE = 3
const MAX_SOURCES = 3

/** Грубый стеммер: срезает частые русские окончания, чтобы «согласие» и «согласия» совпадали. */
function stem(word: string): string {
  const endings = [
    'ами', 'ями', 'ого', 'его', 'ому', 'ему', 'ыми', 'ими',
    'ая', 'яя', 'ое', 'ее', 'ые', 'ие', 'ой', 'ей', 'ый', 'ий',
    'ых', 'их', 'ым', 'им', 'ом', 'ем', 'ов', 'ев', 'ам', 'ям', 'ах', 'ях',
    'а', 'я', 'о', 'е', 'у', 'ю', 'ы', 'и', 'ь', 'й',
  ]
  for (const e of endings) {
    if (word.length - e.length >= 3 && word.endsWith(e)) {
      return word.slice(0, -e.length)
    }
  }
  return word
}

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-zа-яё0-9\s-]/gi, ' ')
    .split(/[\s-]+/)
    .filter((t) => t.length >= 2 && !STOP_WORDS.has(t))
    .map(stem)
}

function normalize(score: number): number {
  return Math.round(Math.min(1, score / 8) * 100) / 100
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

interface RetrievalResult {
  sources: SourceChunk[]
  confidence: number
  answered: boolean
}

/** Заглушка RAG-ретривера: лексическое ранжирование чанков по ключевым словам. */
function runRetrieval(query: string): RetrievalResult {
  const qTokens = tokenize(query)
  const qLower = query.toLowerCase()

  const scored = KB_CHUNKS.map((chunk) => {
    let score = 0

    // Точное/частичное совпадение по ключевым фразам — самый сильный сигнал.
    // Бонус даём только многословным фразам, чтобы одиночные слова не учитывались дважды
    // (иначе «согласие» считалось бы и как фраза, и как токен).
    for (const kw of chunk.keywords) {
      const kl = kw.toLowerCase()
      const isPhrase = kl.includes(' ') || kl.includes('-')
      if (isPhrase && (qLower.includes(kl) || kl.includes(qLower))) score += 4
    }

    // Совпадение по отдельным токенам.
    const kwTokens = new Set(chunk.keywords.flatMap(tokenize))
    const textTokens = new Set(tokenize(chunk.text))
    for (const t of qTokens) {
      if (kwTokens.has(t)) score += 3
      else if (textTokens.has(t)) score += 1
    }

    return { chunk, score }
  })

  scored.sort((a, b) => b.score - a.score)

  const bestRaw = scored[0]?.score ?? 0
  const sources = scored
    .filter((s) => s.score >= RETRIEVAL_MIN_SCORE)
    .slice(0, MAX_SOURCES)
    .map((s) => ({ ...s.chunk, score: normalize(s.score) }))

  return {
    sources,
    confidence: normalize(bestRaw),
    answered: bestRaw >= ANSWER_MIN_SCORE,
  }
}

/** Заглушка LLM: собирает ответ из извлечённых чанков или сообщает, что ответа нет. */
function generateAnswer(query: string, sources: SourceChunk[], answered: boolean): string {
  if (!answered) return notFoundAnswer(query, sources)

  const primary = sources[0]
  const secondary = sources.slice(1).filter((s) => s.score >= 0.5)
  if (secondary.length === 0) return primary.answer

  const extra = secondary.map((s) => `- ${s.cite}: ${s.summary}`).join('\n')
  return `${primary.answer}\n\nДополнительно по теме:\n${extra}`
}

function notFoundAnswer(query: string, sources: SourceChunk[]): string {
  const related = sources.length
    ? `Ближайшие темы в базе знаний:\n${sources.map((s) => `- ${s.cite} — ${s.summary}`).join('\n')}\n\n`
    : ''
  const topics = `В базе знаний доступны разделы:\n${TOPICS.map((t) => `- ${t}`).join('\n')}\n\n`
  return (
    `К сожалению, в базе знаний пока нет ответа на ваш вопрос: «${query}».\n\n` +
    `${related}${topics}` +
    `Попробуйте переформулировать вопрос или уточнить детали (закон, тип документа, ситуацию). ` +
    `Либо нажмите «Передать вопрос юристу», и специалист ПСБ свяжется с вами.`
  )
}

/** Разбивает ответ на «токены» для имитации потокового вывода LLM. */
function tokenizeStream(text: string): string[] {
  return text.match(/\S+\s*/g) ?? [text]
}

export async function* askMock(query: string, signal?: AbortSignal): AsyncGenerator<AskEvent> {
  const start = Date.now()

  // Имитация задержки ретривера.
  await sleep(320 + Math.random() * 480)
  if (signal?.aborted) return

  const { sources, confidence, answered } = runRetrieval(query)
  const status = answered ? 'answered' : 'not_found'
  yield { type: 'retrieval', sources, confidence, status }

  const answer = generateAnswer(query, sources, answered)
  for (const token of tokenizeStream(answer)) {
    if (signal?.aborted) return
    yield { type: 'token', text: token }
    await sleep(8 + Math.random() * 22)
  }

  yield { type: 'done', answer, sources, confidence, status, latencyMs: Date.now() - start }
}
