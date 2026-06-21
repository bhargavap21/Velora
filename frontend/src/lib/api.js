// In dev, Vite proxies /api to the local backend (see vite.config.js), so this is
// empty and paths stay relative. In prod, VITE_API_BASE points straight at the Fly
// backend so SSE streams aren't routed through a Vercel proxy (which buffers/times out).
const API_BASE = import.meta.env.VITE_API_BASE ?? ''

export function apiUrl(path) {
  return `${API_BASE}${path}`
}
