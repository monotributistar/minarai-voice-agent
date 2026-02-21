const DEFAULT_BASE_URL = 'https://api.openai.com/v1'

function fallbackByLang(lang) {
  if (lang.startsWith('en')) {
    return "I'm having trouble reaching my language model right now. I can still summarize my key projects if you want."
  }

  return 'Estoy teniendo problemas para conectar el modelo ahora. Si queres, igual te resumo mis proyectos principales.'
}

export async function generateAssistantReply({ systemPrompt, userText, lang }) {
  const provider = process.env.LLM_PROVIDER || 'openai-compatible'
  const model = process.env.LLM_MODEL || 'gpt-4o-mini'
  const apiKey = process.env.LLM_API_KEY || ''
  const baseUrl = (process.env.LLM_BASE_URL || DEFAULT_BASE_URL).replace(/\/$/, '')

  if (!apiKey) {
    console.warn('[llm] Missing LLM_API_KEY, using fallback response.')
    return fallbackByLang(lang)
  }

  const endpoint = `${baseUrl}/chat/completions`
  const body = {
    model,
    temperature: 0.5,
    max_tokens: 260,
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userText },
    ],
  }

  if (provider !== 'openai-compatible') {
    console.warn(`[llm] Unknown LLM_PROVIDER "${provider}". Falling back to openai-compatible behavior.`)
  }

  for (let attempt = 1; attempt <= 2; attempt += 1) {
    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify(body),
      })

      if (!response.ok) {
        const errText = await response.text()
        throw new Error(`LLM HTTP ${response.status}: ${errText.slice(0, 400)}`)
      }

      const json = await response.json()
      const content = json?.choices?.[0]?.message?.content

      if (typeof content !== 'string' || !content.trim()) {
        throw new Error('LLM response did not include assistant content')
      }

      return content.trim()
    } catch (error) {
      console.error(`[llm] attempt ${attempt} failed:`, error)
      if (attempt === 2) {
        return fallbackByLang(lang)
      }
    }
  }

  return fallbackByLang(lang)
}
