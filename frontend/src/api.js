const BASE = '/api'

export async function parseStrategy(description) {
  const res = await fetch(`${BASE}/parse`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ description }),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function runEpisodes(seeds) {
  const res = await fetch(`${BASE}/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ seeds }),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function runStrategy(strategy) {
  const res = await fetch(`${BASE}/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ strategy }),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function getResults(runId) {
  const res = await fetch(`${BASE}/results/${runId}`)
  if (res.status === 202) return { status: 202 }
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export function openStream(runId) {
  return new EventSource(`${BASE}/stream/${runId}`)
}
