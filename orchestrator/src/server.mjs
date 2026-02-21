import { randomUUID } from 'node:crypto'
import fs from 'node:fs/promises'
import path from 'node:path'
import { spawn } from 'node:child_process'
import { WebSocketServer } from 'ws'
import { getExperience, getProfile, searchProjects } from './cms.mjs'
import { generateAssistantReply } from './llm.mjs'
import { buildGroundingText, buildSystemPrompt, postProcessForTts } from './persona.mjs'

const PORT = Number(process.env.PORT || 8787)

const STT_PROVIDER = (process.env.STT_PROVIDER || 'local').toLowerCase()
const STT_URL = process.env.STT_URL || 'http://stt:9001/transcribe'

const TTS_PROVIDER = (process.env.TTS_PROVIDER || 'local').toLowerCase()
const TTS_URL = process.env.TTS_URL || 'http://tts:9002/speak'

const GROQ_API_KEY = process.env.GROQ_API_KEY || process.env.LLM_API_KEY || ''
const GROQ_STT_URL =
  process.env.GROQ_STT_URL || 'https://api.groq.com/openai/v1/audio/transcriptions'
const GROQ_STT_MODEL = process.env.GROQ_STT_MODEL || 'whisper-large-v3-turbo'
const GROQ_TTS_URL = process.env.GROQ_TTS_URL || 'https://api.groq.com/openai/v1/audio/speech'
const GROQ_TTS_MODEL = process.env.GROQ_TTS_MODEL || 'playai-tts'
const GROQ_TTS_VOICE = process.env.GROQ_TTS_VOICE || 'Fritz-PlayAI'
const GROQ_TTS_INPUT_MAX_CHARS = Number(process.env.GROQ_TTS_INPUT_MAX_CHARS || 180)
const GROQ_TTS_NON_EN_STRATEGY = (process.env.GROQ_TTS_NON_EN_STRATEGY || 'local').toLowerCase()

const MAX_AUDIO_BYTES = Number(process.env.MAX_AUDIO_BYTES || 8 * 1024 * 1024)
const MAX_AUDIO_SECONDS = Number(process.env.MAX_AUDIO_SECONDS || 30)
const TMP_DIR = process.env.TMP_DIR || '/tmp'
const ALLOWED_ORIGINS = (process.env.ALLOWED_ORIGINS || '')
  .split(',')
  .map((s) => s.trim())
  .filter(Boolean)

const wss = new WebSocketServer({ port: PORT })
const connectionState = new WeakMap()

function nowMs() {
  return Date.now()
}

function sendJson(ws, payload) {
  if (ws.readyState === ws.OPEN) {
    ws.send(JSON.stringify(payload))
  }
}

function logStep(requestId, step, ts) {
  console.log(`[${requestId}] ${step}=${ts}`)
}

function execCmd(command, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args)
    let stderr = ''

    child.stderr.on('data', (chunk) => {
      stderr += chunk.toString()
    })

    child.on('error', reject)
    child.on('close', (code) => {
      if (code === 0) {
        resolve()
      } else {
        reject(new Error(`${command} exited ${code}: ${stderr}`))
      }
    })
  })
}

async function getAudioDurationSeconds(filePath) {
  return new Promise((resolve, reject) => {
    const args = [
      '-v',
      'error',
      '-show_entries',
      'format=duration',
      '-of',
      'default=nokey=1:noprint_wrappers=1',
      filePath,
    ]
    const child = spawn('ffprobe', args)
    let output = ''
    let stderr = ''

    child.stdout.on('data', (chunk) => {
      output += chunk.toString()
    })

    child.stderr.on('data', (chunk) => {
      stderr += chunk.toString()
    })

    child.on('error', reject)
    child.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`ffprobe exited ${code}: ${stderr}`))
        return
      }

      const parsed = Number(output.trim())
      if (Number.isNaN(parsed)) {
        reject(new Error(`Invalid ffprobe duration output: ${output}`))
        return
      }

      resolve(parsed)
    })
  })
}

async function convertToWav(inputPath, outputPath) {
  await execCmd('ffmpeg', [
    '-y',
    '-i',
    inputPath,
    '-ac',
    '1',
    '-ar',
    '16000',
    '-c:a',
    'pcm_s16le',
    outputPath,
  ])
}

async function transcribeWavLocal(buffer, requestId) {
  const blob = new Blob([buffer], { type: 'audio/wav' })
  const form = new FormData()
  form.append('file', blob, `${requestId}.wav`)

  const response = await fetch(STT_URL, {
    method: 'POST',
    body: form,
  })

  if (!response.ok) {
    const errorText = await response.text()
    throw new Error(`STT error ${response.status}: ${errorText.slice(0, 300)}`)
  }

  return response.json()
}

async function transcribeWavGroq(buffer, requestId, lang) {
  if (!GROQ_API_KEY) {
    throw new Error('STT_PROVIDER=groq but GROQ_API_KEY (or LLM_API_KEY) is missing')
  }

  const blob = new Blob([buffer], { type: 'audio/wav' })
  const form = new FormData()
  form.append('file', blob, `${requestId}.wav`)
  form.append('model', GROQ_STT_MODEL)
  form.append('response_format', 'verbose_json')
  form.append('language', lang.startsWith('en') ? 'en' : 'es')

  const response = await fetch(GROQ_STT_URL, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${GROQ_API_KEY}`,
    },
    body: form,
  })

  if (!response.ok) {
    const errorText = await response.text()
    throw new Error(`Groq STT error ${response.status}: ${errorText.slice(0, 400)}`)
  }

  const json = await response.json()
  const segments =
    Array.isArray(json?.segments) && json.segments.length > 0
      ? json.segments.map((segment) => ({
          start: Number(segment.start || 0),
          end: Number(segment.end || 0),
          text: String(segment.text || '').trim(),
        }))
      : []

  return {
    text: String(json?.text || '').trim(),
    language: String(json?.language || ''),
    segments,
  }
}

async function transcribeWav(buffer, requestId, lang = 'es-AR') {
  if (STT_PROVIDER === 'groq') {
    return transcribeWavGroq(buffer, requestId, lang)
  }
  return transcribeWavLocal(buffer, requestId)
}

async function synthesizeSpeechLocal(text, lang) {
  const voice = lang.startsWith('en') ? 'en' : 'es'
  const response = await fetch(TTS_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ text, voice }),
  })

  if (!response.ok) {
    const err = await response.text()
    throw new Error(`TTS error ${response.status}: ${err.slice(0, 300)}`)
  }

  return Buffer.from(await response.arrayBuffer())
}

async function synthesizeSpeechGroq(text, lang) {
  if (!GROQ_API_KEY) {
    throw new Error('TTS_PROVIDER=groq but GROQ_API_KEY (or LLM_API_KEY) is missing')
  }

  const isEnglish = lang.startsWith('en')
  const fallbackLocal = !isEnglish && GROQ_TTS_NON_EN_STRATEGY === 'local'
  if (fallbackLocal) {
    console.warn(
      `[tts] Non-English request (${lang}) with Groq TTS. Falling back to local TTS because GROQ_TTS_NON_EN_STRATEGY=local.`,
    )
    return synthesizeSpeechLocal(text, lang)
  }

  const input = text.slice(0, Math.max(1, GROQ_TTS_INPUT_MAX_CHARS)).trim()
  const response = await fetch(GROQ_TTS_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${GROQ_API_KEY}`,
    },
    body: JSON.stringify({
      model: GROQ_TTS_MODEL,
      voice: GROQ_TTS_VOICE,
      input,
      response_format: 'wav',
    }),
  })

  if (!response.ok) {
    const err = await response.text()
    throw new Error(`Groq TTS error ${response.status}: ${err.slice(0, 400)}`)
  }

  return Buffer.from(await response.arrayBuffer())
}

async function synthesizeSpeech(text, lang) {
  if (TTS_PROVIDER === 'groq') {
    return synthesizeSpeechGroq(text, lang)
  }
  return synthesizeSpeechLocal(text, lang)
}

function shouldGroundProjects(query = '') {
  const q = query.toLowerCase()
  return /(project|proyecto|stack|skill|habilidad|trabajo|experiencia|scan|3d|accesibilidad|accessibility)/.test(
    q,
  )
}

async function buildGrounding(lang, userText) {
  const [profile, experience] = await Promise.all([getProfile(lang), getExperience(lang)])
  const projects = shouldGroundProjects(userText) ? await searchProjects(userText, lang) : []

  return buildGroundingText({ profile, projects, experience })
}

async function processAudioRequest(ws, state, payload) {
  const requestId = payload.requestId || randomUUID()
  const lang = payload.lang || 'es-AR'

  const stamps = {
    audio_received: nowMs(),
    ffmpeg_done: null,
    stt_done: null,
    llm_done: null,
    tts_done: null,
    ws_sent: null,
  }

  const inputPath = path.join(TMP_DIR, `${requestId}.input`)
  const wavPath = path.join(TMP_DIR, `${requestId}.wav`)

  try {
    sendJson(ws, { type: 'state', requestId, state: 'uploading' })

    await fs.writeFile(inputPath, payload.binary)

    const duration = await getAudioDurationSeconds(inputPath)
    if (duration > MAX_AUDIO_SECONDS) {
      throw new Error(`Audio too long (${duration.toFixed(2)}s). Max is ${MAX_AUDIO_SECONDS}s`)
    }

    sendJson(ws, { type: 'state', requestId, state: 'transcribing' })
    await convertToWav(inputPath, wavPath)
    stamps.ffmpeg_done = nowMs()
    logStep(requestId, 'ffmpeg_done', stamps.ffmpeg_done)

    const wavBuffer = await fs.readFile(wavPath)
    const sttResult = await transcribeWav(wavBuffer, requestId, lang)
    stamps.stt_done = nowMs()
    logStep(requestId, 'stt_done', stamps.stt_done)

    const userText = (sttResult?.text || '').trim()

    sendJson(ws, {
      type: 'transcript',
      requestId,
      role: 'user',
      text: userText,
      isFinal: true,
    })

    sendJson(ws, { type: 'state', requestId, state: 'thinking' })

    const grounding = await buildGrounding(lang, userText)
    const systemPrompt = buildSystemPrompt({ lang, grounding })
    const assistantTextRaw = await generateAssistantReply({
      systemPrompt,
      userText,
      lang,
    })

    const assistantText = postProcessForTts(assistantTextRaw)
    stamps.llm_done = nowMs()
    logStep(requestId, 'llm_done', stamps.llm_done)

    sendJson(ws, {
      type: 'transcript',
      requestId,
      role: 'assistant',
      text: assistantText,
      isFinal: true,
    })

    sendJson(ws, { type: 'state', requestId, state: 'speaking' })

    const audioBytes = await synthesizeSpeech(assistantText, lang)
    stamps.tts_done = nowMs()
    logStep(requestId, 'tts_done', stamps.tts_done)

    sendJson(ws, {
      type: 'audio_meta',
      requestId,
      mime: 'audio/wav',
    })
    ws.send(audioBytes, { binary: true })
    stamps.ws_sent = nowMs()
    logStep(requestId, 'ws_sent', stamps.ws_sent)

    sendJson(ws, { type: 'state', requestId, state: 'idle' })
  } finally {
    await Promise.allSettled([fs.rm(inputPath, { force: true }), fs.rm(wavPath, { force: true })])
    state.busy = false
    state.pendingMeta = null
  }
}

wss.on('connection', (ws, req) => {
  const origin = req.headers.origin || ''
  if (ALLOWED_ORIGINS.length > 0 && !ALLOWED_ORIGINS.includes(origin)) {
    ws.close(1008, 'Origin not allowed')
    return
  }

  const state = {
    busy: false,
    pendingMeta: null,
    isAlive: true,
  }
  connectionState.set(ws, state)

  ws.on('pong', () => {
    state.isAlive = true
  })

  ws.on('message', async (data, isBinary) => {
    try {
      if (!isBinary) {
        const text = data.toString()
        const message = JSON.parse(text)

        if (message.type === 'ping') {
          sendJson(ws, { type: 'pong', ts: Date.now() })
          return
        }

        if (message.type === 'cancel') {
          sendJson(ws, { type: 'state', state: 'idle' })
          state.pendingMeta = null
          return
        }

        if (message.type !== 'audio_meta') {
          sendJson(ws, { type: 'error', message: `Unsupported message type: ${message.type}` })
          return
        }

        if (state.busy) {
          sendJson(ws, {
            type: 'error',
            requestId: message.requestId,
            message: 'Connection is busy with another request.',
          })
          return
        }

        state.pendingMeta = {
          requestId: message.requestId || randomUUID(),
          lang: message.lang || 'es-AR',
          mime: message.mime || 'application/octet-stream',
        }
        return
      }

      if (!state.pendingMeta) {
        sendJson(ws, {
          type: 'error',
          message: 'Binary audio received without preceding audio_meta frame.',
        })
        return
      }

      const binary = Buffer.from(data)
      if (binary.byteLength > MAX_AUDIO_BYTES) {
        sendJson(ws, {
          type: 'error',
          requestId: state.pendingMeta.requestId,
          message: `Audio payload too large. Max bytes: ${MAX_AUDIO_BYTES}`,
        })
        state.pendingMeta = null
        return
      }

      if (state.busy) {
        sendJson(ws, {
          type: 'error',
          requestId: state.pendingMeta.requestId,
          message: 'Connection is busy with another request.',
        })
        state.pendingMeta = null
        return
      }

      state.busy = true

      const requestId = state.pendingMeta.requestId
      logStep(requestId, 'audio_received', nowMs())

      await processAudioRequest(ws, state, {
        ...state.pendingMeta,
        binary,
      })
    } catch (error) {
      console.error('[ws] processing error:', error)
      const requestId = state.pendingMeta?.requestId
      sendJson(ws, {
        type: 'error',
        requestId,
        message: error instanceof Error ? error.message : 'Unknown error',
      })
      sendJson(ws, { type: 'state', requestId, state: 'error' })
      state.busy = false
      state.pendingMeta = null
    }
  })

  ws.on('close', () => {
    connectionState.delete(ws)
  })
})

const pingInterval = setInterval(() => {
  wss.clients.forEach((ws) => {
    const state = connectionState.get(ws)
    if (!state) return

    if (!state.isAlive) {
      ws.terminate()
      connectionState.delete(ws)
      return
    }

    state.isAlive = false
    ws.ping()
  })
}, 20_000)

wss.on('close', () => {
  clearInterval(pingInterval)
})

console.log(`[voice-orchestrator] listening on ws://0.0.0.0:${PORT}`)
