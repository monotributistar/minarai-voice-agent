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
const F5_TTS_URL = process.env.F5_TTS_URL || 'http://tts-f5:9003/speak'
const F5_TTS_TIMEOUT_MS = Number(process.env.F5_TTS_TIMEOUT_MS || 120000)
const TTS_NODE_TIMEOUT_MS = Number(process.env.TTS_NODE_TIMEOUT_MS || 15000)
const TTS_PROVIDER_REGISTRY = (process.env.TTS_PROVIDER_REGISTRY || 'local,groq,f5')
  .split(',')
  .map((item) => item.trim().toLowerCase())
  .filter(Boolean)

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
const providerState = new Map()

for (const provider of new Set([...TTS_PROVIDER_REGISTRY, 'local', 'groq', 'f5'])) {
  providerState.set(provider, {
    id: provider,
    enabled: provider === TTS_PROVIDER || TTS_PROVIDER_REGISTRY.includes(provider),
  })
}

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

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value))
}

function toNumber(value) {
  const num = Number(value)
  return Number.isFinite(num) ? num : null
}

function withTimeoutSignal(timeoutMs = 5000) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), Math.max(500, timeoutMs))
  return { signal: controller.signal, cancel: () => clearTimeout(timer) }
}

function getProviderHealthUrl(provider) {
  if (provider === 'f5') return F5_TTS_URL.replace(/\/speak$/, '/health')
  return TTS_URL.replace(/\/speak$/, '/health')
}

function getProviderSpeakUrl(provider) {
  return provider === 'f5' ? F5_TTS_URL : TTS_URL
}

async function fetchJson(url, timeoutMs = 5000) {
  const { signal, cancel } = withTimeoutSignal(timeoutMs)
  try {
    const response = await fetch(url, { signal })
    if (!response.ok) {
      const text = await response.text()
      throw new Error(`HTTP ${response.status}: ${text.slice(0, 240)}`)
    }
    return await response.json()
  } finally {
    cancel()
  }
}

function normalizeTtsNodeConfig(raw, lang) {
  const defaultVoice = lang.startsWith('en') ? 'en' : 'es'
  const speed = clamp(toNumber(raw?.prosody?.speed) ?? 1, 0.7, 1.4)
  const lengthScale = clamp(1 / speed, 0.7, 1.6)
  const pitch = clamp(toNumber(raw?.prosody?.pitch) ?? 0, -1, 1)
  const energy = clamp(toNumber(raw?.prosody?.energy) ?? 0.8, 0, 1)
  const pauseMs = clamp(toNumber(raw?.prosody?.pause_ms) ?? 120, 0, 500)
  const temperature = clamp(toNumber(raw?.generation?.temperature) ?? 0.6, 0, 1)
  const maxChars = clamp(
    Math.floor(toNumber(raw?.constraints?.max_chars) ?? GROQ_TTS_INPUT_MAX_CHARS),
    60,
    900,
  )

  return {
    provider: raw?.provider ? String(raw.provider).toLowerCase() : null,
    instanceId: raw?.instance_id ? String(raw.instance_id) : null,
    voice: String(raw?.voice || defaultVoice),
    style: String(raw?.style || 'friendly_direct'),
    prosody: {
      speed,
      pitch,
      energy,
      pauseMs,
      lengthScale,
      noiseScale: clamp(0.3 + (1 - energy) * 0.4, 0.1, 1),
      noiseW: clamp(0.6 - pitch * 0.15, 0.2, 1),
      sentenceSilence: clamp(pauseMs / 1000, 0, 1),
    },
    generation: {
      temperature,
      stability: clamp(toNumber(raw?.generation?.stability) ?? 0.75, 0, 1),
      seed: Math.floor(toNumber(raw?.generation?.seed) ?? 42),
      model: String(raw?.generation?.model || GROQ_TTS_MODEL),
      providerVoice: String(raw?.generation?.provider_voice || GROQ_TTS_VOICE),
    },
    audio: {
      format: String(raw?.audio?.format || 'wav'),
      sampleRate: Math.floor(toNumber(raw?.audio?.sample_rate) ?? 22050),
    },
    constraints: {
      maxChars,
    },
  }
}

async function getProviderStatus() {
  const statuses = []
  const providers = [...providerState.keys()]
  for (const provider of providers) {
    const item = providerState.get(provider)
    let healthy = false
    let info = null
    try {
      const health = await fetchJson(getProviderHealthUrl(provider), 2500)
      healthy = Boolean(health?.ok)
      info = health
    } catch (error) {
      healthy = false
      info = { error: error instanceof Error ? error.message : String(error) }
    }
    statuses.push({
      id: provider,
      enabled: Boolean(item?.enabled),
      healthy,
      info,
    })
  }
  return statuses
}

async function resolveActiveProvider(preferredProvider = null) {
  const normalizedPreferred = preferredProvider ? String(preferredProvider).toLowerCase() : null
  const statuses = await getProviderStatus()
  if (normalizedPreferred) {
    const preferred = statuses.find((item) => item.id === normalizedPreferred)
    if (preferred?.enabled) {
      return { provider: preferred.id, statuses }
    }
  }

  const firstHealthyEnabled = statuses.find((item) => item.enabled && item.healthy)
  if (firstHealthyEnabled) {
    return { provider: firstHealthyEnabled.id, statuses }
  }

  const firstEnabled = statuses.find((item) => item.enabled)
  if (firstEnabled) {
    return { provider: firstEnabled.id, statuses }
  }

  throw new Error('No active TTS providers enabled. Enable at least one provider in debug panel.')
}

function getNodeTtsBaseUrl(provider = TTS_PROVIDER) {
  if (provider === 'f5') return F5_TTS_URL.replace(/\/speak$/, '')
  return TTS_URL.replace(/\/speak$/, '')
}

function isNodeTtsProvider(provider = TTS_PROVIDER) {
  return provider === 'local' || provider === 'f5' || provider === 'groq'
}

function getNodeProviderRoute(provider = TTS_PROVIDER) {
  if (provider === 'f5') return 'f5'
  if (provider === 'local' || provider === 'groq') return 'tts'
  return 'tts'
}

async function fetchJsonWithMethod(url, method = 'GET', payload = null, timeoutMs = 5000) {
  const { signal, cancel } = withTimeoutSignal(timeoutMs)
  try {
    const response = await fetch(url, {
      method,
      headers: payload ? { 'Content-Type': 'application/json' } : undefined,
      body: payload ? JSON.stringify(payload) : undefined,
      signal,
    })
    if (!response.ok) {
      const text = await response.text()
      throw new Error(`HTTP ${response.status}: ${text.slice(0, 300)}`)
    }
    return await response.json()
  } finally {
    cancel()
  }
}

async function getNodeConfig(provider = TTS_PROVIDER) {
  if (!isNodeTtsProvider(provider)) {
    throw new Error(`Provider ${provider} does not expose node config endpoints`)
  }
  const baseUrl = getNodeTtsBaseUrl(provider)
  return fetchJsonWithMethod(`${baseUrl}/config`, 'GET', null, 6000)
}

async function updateNodeConfig(provider = TTS_PROVIDER, configPayload = {}) {
  if (!isNodeTtsProvider(provider)) {
    throw new Error(`Provider ${provider} does not expose node config endpoints`)
  }
  const baseUrl = getNodeTtsBaseUrl(provider)
  return fetchJsonWithMethod(`${baseUrl}/config`, 'PUT', configPayload, 6000)
}

async function setNodeEphemeral(provider = TTS_PROVIDER, ephemeralPayload = {}) {
  if (!isNodeTtsProvider(provider)) {
    throw new Error(`Provider ${provider} does not expose node config endpoints`)
  }
  const baseUrl = getNodeTtsBaseUrl(provider)
  return fetchJsonWithMethod(`${baseUrl}/config/ephemeral`, 'POST', ephemeralPayload, 6000)
}

async function clearNodeEphemeral(provider = TTS_PROVIDER) {
  if (!isNodeTtsProvider(provider)) {
    throw new Error(`Provider ${provider} does not expose node config endpoints`)
  }
  const baseUrl = getNodeTtsBaseUrl(provider)
  return fetchJsonWithMethod(`${baseUrl}/config/ephemeral`, 'DELETE', null, 6000)
}

async function getTtsDebugSnapshot(ttsConfig) {
  const statuses = await getProviderStatus()
  const snapshot = {
    ok: true,
    ttsProvider: TTS_PROVIDER,
    ttsProviders: statuses,
    ttsService: {
      healthy: false,
      details: null,
    },
    ttsInstances: [],
    activeTtsConfig: {
      instance_id: ttsConfig.instanceId || undefined,
      voice: ttsConfig.voice,
      style: ttsConfig.style,
      prosody: {
        speed: ttsConfig.prosody.speed,
        pitch: ttsConfig.prosody.pitch,
        energy: ttsConfig.prosody.energy,
        pause_ms: ttsConfig.prosody.pauseMs,
      },
      generation: {
        temperature: ttsConfig.generation.temperature,
        stability: ttsConfig.generation.stability,
        seed: ttsConfig.generation.seed,
        model: ttsConfig.generation.model,
        provider_voice: ttsConfig.generation.providerVoice,
      },
      audio: {
        format: ttsConfig.audio.format,
        sample_rate: ttsConfig.audio.sampleRate,
      },
      constraints: {
        max_chars: ttsConfig.constraints.maxChars,
      },
    },
  }

  const enabledProvider = statuses.find((item) => item.enabled)
  const effectiveProvider = enabledProvider ? enabledProvider.id : TTS_PROVIDER

  if (!isNodeTtsProvider(effectiveProvider)) {
    snapshot.ttsService.healthy = true
    snapshot.ttsService.details = {
      info: `provider=${effectiveProvider} (local instance list not required)`,
    }
    return snapshot
  }

  try {
    const baseUrl = getNodeTtsBaseUrl(effectiveProvider)
    const [health, instancesResponse, configResponse] = await Promise.all([
      fetchJson(`${baseUrl}/health`, 6000),
      fetchJson(`${baseUrl}/instances`, 6000),
      fetchJson(`${baseUrl}/config`, 6000),
    ])

    snapshot.ttsService.healthy = Boolean(health?.ok)
    snapshot.ttsService.details = health
    snapshot.ttsInstances = Array.isArray(instancesResponse?.instances)
      ? instancesResponse.instances
      : []
    snapshot.nodeConfig = configResponse?.config || null
    snapshot.nodeRoute = getNodeProviderRoute(effectiveProvider)
    return snapshot
  } catch (error) {
    snapshot.ok = false
    snapshot.ttsService.healthy = false
    snapshot.error = error instanceof Error ? error.message : 'debug snapshot failed'
    return snapshot
  }
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

async function synthesizeSpeechLocal(text, lang, ttsConfig) {
  const voice = ttsConfig.voice.startsWith('en') ? 'en' : lang.startsWith('en') ? 'en' : 'es'
  const response = await fetch(TTS_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      text,
      voice,
      instance_id: ttsConfig.instanceId,
      length_scale: ttsConfig.prosody.lengthScale,
      noise_scale: ttsConfig.prosody.noiseScale,
      noise_w: ttsConfig.prosody.noiseW,
      sentence_silence: ttsConfig.prosody.sentenceSilence,
    }),
  })

  if (!response.ok) {
    const err = await response.text()
    throw new Error(`TTS error ${response.status}: ${err.slice(0, 300)}`)
  }

  return Buffer.from(await response.arrayBuffer())
}

async function synthesizeSpeechGroq(text, lang, ttsConfig) {
  if (!GROQ_API_KEY) {
    throw new Error('TTS_PROVIDER=groq but GROQ_API_KEY (or LLM_API_KEY) is missing')
  }

  const isEnglish = lang.startsWith('en')
  const fallbackLocal = !isEnglish && GROQ_TTS_NON_EN_STRATEGY === 'local'
  if (fallbackLocal) {
    console.warn(
      `[tts] Non-English request (${lang}) with Groq TTS. Falling back to local TTS because GROQ_TTS_NON_EN_STRATEGY=local.`,
    )
    return synthesizeSpeechLocal(text, lang, ttsConfig)
  }

  const input = text.slice(0, Math.max(1, ttsConfig.constraints.maxChars)).trim()
  const response = await fetch(GROQ_TTS_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${GROQ_API_KEY}`,
    },
    body: JSON.stringify({
      model: ttsConfig.generation.model || GROQ_TTS_MODEL,
      voice: ttsConfig.generation.providerVoice || GROQ_TTS_VOICE,
      input,
      response_format: ttsConfig.audio.format || 'wav',
    }),
  })

  if (!response.ok) {
    const err = await response.text()
    throw new Error(`Groq TTS error ${response.status}: ${err.slice(0, 400)}`)
  }

  return Buffer.from(await response.arrayBuffer())
}

async function synthesizeSpeechNode(
  speakUrl,
  nodeProvider,
  text,
  lang,
  ttsConfig,
  timeoutMs = TTS_NODE_TIMEOUT_MS,
) {
  const voice = ttsConfig.voice.startsWith('en') ? 'en' : lang.startsWith('en') ? 'en' : 'es'
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), Math.max(3000, timeoutMs))

  try {
    const response = await fetch(speakUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text,
        lang,
        provider: nodeProvider,
        voice,
        instance_id: ttsConfig.instanceId || null,
        style: ttsConfig.style,
        prosody: {
          speed: ttsConfig.prosody.speed,
          pitch: ttsConfig.prosody.pitch,
          energy: ttsConfig.prosody.energy,
          pause_ms: ttsConfig.prosody.pauseMs,
        },
        generation: {
          temperature: ttsConfig.generation.temperature,
          stability: ttsConfig.generation.stability,
          seed: ttsConfig.generation.seed,
          model: ttsConfig.generation.model,
          provider_voice: ttsConfig.generation.providerVoice,
        },
        audio: {
          format: ttsConfig.audio.format,
          sample_rate: ttsConfig.audio.sampleRate,
        },
        constraints: {
          max_chars: ttsConfig.constraints.maxChars,
        },
        length_scale: ttsConfig.prosody.lengthScale,
        noise_scale: ttsConfig.prosody.noiseScale,
        noise_w: ttsConfig.prosody.noiseW,
        sentence_silence: ttsConfig.prosody.sentenceSilence,
        speed: ttsConfig.prosody.speed,
      }),
      signal: controller.signal,
    })

    if (!response.ok) {
      const err = await response.text()
      throw new Error(`TTS node error ${response.status}: ${err.slice(0, 500)}`)
    }

    return Buffer.from(await response.arrayBuffer())
  } finally {
    clearTimeout(timeout)
  }
}

async function synthesizeSpeech(text, lang, ttsConfig) {
  const { provider } = await resolveActiveProvider(ttsConfig.provider)

  if (isNodeTtsProvider(provider)) {
    const speakUrl = getProviderSpeakUrl(provider)
    const timeoutMs = provider === 'f5' ? F5_TTS_TIMEOUT_MS : TTS_NODE_TIMEOUT_MS
    return synthesizeSpeechNode(speakUrl, provider, text, lang, ttsConfig, timeoutMs)
  }

  if (provider === 'groq') {
    return synthesizeSpeechGroq(text, lang, ttsConfig)
  }
  return synthesizeSpeechLocal(text, lang, ttsConfig)
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
  const ttsNodeConfig = normalizeTtsNodeConfig(payload.tts, lang)

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

    const assistantText = postProcessForTts(assistantTextRaw).slice(0, ttsNodeConfig.constraints.maxChars)
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

    const audioBytes = await synthesizeSpeech(assistantText, lang, ttsNodeConfig)
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

        if (message.type === 'debug_ping') {
          const debugLang = message.lang || 'es-AR'
          const debugConfig = normalizeTtsNodeConfig(message.tts || null, debugLang)
          const debugStatus = await getTtsDebugSnapshot(debugConfig)
          sendJson(ws, {
            type: 'debug_status',
            requestId: message.requestId || null,
            data: debugStatus,
          })
          return
        }

        if (message.type === 'provider_set') {
          const providerId = String(message.provider || '').toLowerCase()
          if (!providerState.has(providerId)) {
            sendJson(ws, { type: 'error', message: `Unknown provider: ${providerId}` })
            return
          }
          const enabled = Boolean(message.enabled)
          providerState.set(providerId, { id: providerId, enabled })
          const debugConfig = normalizeTtsNodeConfig(message.tts || null, message.lang || 'es-AR')
          const debugStatus = await getTtsDebugSnapshot(debugConfig)
          sendJson(ws, { type: 'providers_status', data: debugStatus.ttsProviders })
          sendJson(ws, { type: 'debug_status', data: debugStatus })
          return
        }

        if (message.type === 'node_config_get') {
          const providerId = String(message.provider || TTS_PROVIDER).toLowerCase()
          try {
            const data = await getNodeConfig(providerId)
            sendJson(ws, { type: 'node_config_status', provider: providerId, data })
          } catch (error) {
            sendJson(ws, {
              type: 'error',
              message: error instanceof Error ? error.message : 'node_config_get failed',
            })
          }
          return
        }

        if (message.type === 'node_config_set') {
          const providerId = String(message.provider || TTS_PROVIDER).toLowerCase()
          try {
            const data = await updateNodeConfig(providerId, message.config || {})
            sendJson(ws, { type: 'node_config_status', provider: providerId, data })
            const debugConfig = normalizeTtsNodeConfig(message.tts || null, message.lang || 'es-AR')
            const debugStatus = await getTtsDebugSnapshot(debugConfig)
            sendJson(ws, { type: 'debug_status', data: debugStatus })
          } catch (error) {
            sendJson(ws, {
              type: 'error',
              message: error instanceof Error ? error.message : 'node_config_set failed',
            })
          }
          return
        }

        if (message.type === 'node_ephemeral_set') {
          const providerId = String(message.provider || TTS_PROVIDER).toLowerCase()
          try {
            const payload = {
              instance: message.instance || {},
              ttl_seconds: Number(message.ttl_seconds || 900),
            }
            const data = await setNodeEphemeral(providerId, payload)
            sendJson(ws, { type: 'node_config_status', provider: providerId, data })
            const debugConfig = normalizeTtsNodeConfig(message.tts || null, message.lang || 'es-AR')
            const debugStatus = await getTtsDebugSnapshot(debugConfig)
            sendJson(ws, { type: 'debug_status', data: debugStatus })
          } catch (error) {
            sendJson(ws, {
              type: 'error',
              message: error instanceof Error ? error.message : 'node_ephemeral_set failed',
            })
          }
          return
        }

        if (message.type === 'node_ephemeral_clear') {
          const providerId = String(message.provider || TTS_PROVIDER).toLowerCase()
          try {
            const data = await clearNodeEphemeral(providerId)
            sendJson(ws, { type: 'node_config_status', provider: providerId, data })
            const debugConfig = normalizeTtsNodeConfig(message.tts || null, message.lang || 'es-AR')
            const debugStatus = await getTtsDebugSnapshot(debugConfig)
            sendJson(ws, { type: 'debug_status', data: debugStatus })
          } catch (error) {
            sendJson(ws, {
              type: 'error',
              message: error instanceof Error ? error.message : 'node_ephemeral_clear failed',
            })
          }
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
          tts: message.tts || null,
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
