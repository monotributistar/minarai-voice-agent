import fs from 'node:fs/promises'
import path from 'node:path'

const CMS_DIR = process.env.CMS_DIR || '/app/cms'

let cache = null

async function loadContent() {
  if (cache) return cache

  const [esRaw, enRaw] = await Promise.all([
    fs.readFile(path.join(CMS_DIR, 'content.es.json'), 'utf8'),
    fs.readFile(path.join(CMS_DIR, 'content.en.json'), 'utf8'),
  ])

  cache = {
    es: JSON.parse(esRaw),
    en: JSON.parse(enRaw),
  }

  return cache
}

function normalizeLang(lang) {
  return lang.startsWith('en') ? 'en' : 'es'
}

export async function getProfile(lang) {
  const data = await loadContent()
  const key = normalizeLang(lang)
  const cms = data[key]

  return {
    siteName: cms.meta.siteName,
    contactEmail: cms.meta.contactEmail,
    hero: cms.hero,
    cv: {
      title: cms.cv.title,
      subtitle: cms.cv.subtitle,
      summary: cms.cv.summary.slice(0, 3),
      highlights: cms.cv.highlights,
    },
    skills: cms.skills.tags.slice(0, 20),
    social: cms.social.links,
  }
}

export async function searchProjects(query, lang) {
  const data = await loadContent()
  const key = normalizeLang(lang)
  const cms = data[key]

  const q = (query || '').toLowerCase().trim()
  const terms = q.split(/\s+/).filter(Boolean)

  const scored = cms.projects.items.map((project) => {
    const bag = `${project.title} ${project.summary} ${project.stack} ${(project.roles || []).join(' ')}`.toLowerCase()
    const score = terms.length === 0 ? 1 : terms.reduce((acc, term) => acc + (bag.includes(term) ? 1 : 0), 0)
    return { project, score }
  })

  return scored
    .filter((item) => item.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, 5)
    .map(({ project }) => ({
      id: project.id,
      title: project.title,
      summary: project.summary,
      stack: project.stack,
      roles: project.roles,
    }))
}

export async function getExperience(lang) {
  const data = await loadContent()
  const key = normalizeLang(lang)
  const cms = data[key]

  const timeline = [...cms.activities.timeline]
    .sort((a, b) => (a.start < b.start ? 1 : -1))
    .slice(0, 6)
    .map((item) => ({
      kind: item.kind,
      title: item.title,
      description: item.description,
      start: item.start,
      end: item.end || null,
      tags: item.tags,
    }))

  return {
    activitiesTitle: cms.activities.title,
    timeline,
  }
}
