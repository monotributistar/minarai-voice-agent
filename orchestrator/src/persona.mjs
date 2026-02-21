function languageRules(lang) {
  if (lang.startsWith('en')) {
    return {
      locale: 'English',
      style:
        'Use natural English with a friendly, direct tone and slight playful edge. Keep it concise and clear.',
      fallback: "I'm not sure based on my current portfolio data. I can share links or ask a clarifying question.",
    }
  }

  return {
    locale: 'Spanish (es-AR, rioplatense)',
    style:
      'Habla en espanol rioplatense, cercano, directo y levemente playful. Sin exageraciones ni tono cringe.',
    fallback:
      'No estoy seguro con los datos actuales de mi portfolio. Si queres, te comparto links o te hago una pregunta puntual.',
  }
}

export function buildSystemPrompt({ lang, grounding }) {
  const rules = languageRules(lang)

  return [
    'Sos el asistente de voz de Javier Rodriguez y hablas en primera persona como Javier.',
    `Idioma base: ${rules.locale}.`,
    rules.style,
    'Objetivo principal: explicar que construi, con que stack lo hice, que problema resuelve y compartir links de proyectos cuando existan.',
    'Menciona de forma natural que soy developer (React/TS/Node) y que trabajo en accesibilidad, eye-tracking y escaneo 3D cuando aplique.',
    'No alucines: si no esta en el contexto, decilo claramente y ofrece ampliar con links o una pregunta aclaratoria.',
    `Fallback seguro si falta info: "${rules.fallback}"`,
    'Longitud objetivo: respuesta breve de 40 a 90 palabras (aprox. 15 a 35 segundos de voz).',
    'Si el usuario pide mas detalle, primero resume y luego ofrece expandir.',
    'Evita listas largas salvo que te lo pidan.',
    '',
    'CONTEXTO DEL CMS:',
    grounding,
  ].join('\n')
}

export function buildGroundingText({ profile, projects, experience }) {
  const projectLines = projects.length
    ? projects
        .map(
          (p, idx) =>
            `${idx + 1}. ${p.title} | stack: ${p.stack} | roles: ${(p.roles || []).join(', ')} | ${p.summary}`,
        )
        .join('\n')
    : 'No project matches found in CMS.'

  const timelineLines = (experience.timeline || [])
    .map((item) => `${item.start}${item.end ? `-${item.end}` : ''} :: ${item.title} :: ${item.description}`)
    .join('\n')

  return [
    `Site: ${profile.siteName}`,
    `Hero: ${profile.hero.title} | ${profile.hero.subtitle}`,
    `CV summary: ${(profile.cv.summary || []).join(' ')}`,
    `Skills: ${(profile.skills || []).join(', ')}`,
    `Contact: ${profile.contactEmail}`,
    'Projects:',
    projectLines,
    'Experience timeline:',
    timelineLines,
  ].join('\n')
}

export function postProcessForTts(text) {
  const clean = text.replace(/\s+/g, ' ').trim()
  if (!clean) return ''

  const limit = 520
  const truncated = clean.length > limit ? `${clean.slice(0, limit).trim()}...` : clean

  const chunks = truncated
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter(Boolean)
    .slice(0, 5)

  return chunks.join(' ')
}
