import { Link, useLocation } from 'react-router-dom'
import { cn } from '@/lib/utils'

const LINKS = [
  { to: '/showdown', label: 'Showdown' },
  { to: '/proof', label: 'Proof' },
  { to: '/execution-demo', label: 'Live Demo' },
  { to: '/sandbox', label: 'Sandbox' },
  { to: '/rft', label: 'RFT' },
]

export default function Navbar() {
  const { pathname } = useLocation()
  return (
    <nav
      className="sticky top-0 z-50 flex h-16 items-center justify-between border-b border-[#2e3038] bg-[#000000]/95 px-6 backdrop-blur-sm md:px-12"
      style={{ fontFamily: "'Inter', sans-serif" }}
    >
      <Link
        to="/"
        className="text-[22px] text-white"
        style={{ fontFamily: "'Playfair Display', Georgia, serif", fontStyle: 'italic', fontWeight: 400 }}
      >
        Velora
      </Link>

      <div className="flex items-center gap-6 md:gap-10">
        {LINKS.map(l => (
          <Link
            key={l.to}
            to={l.to}
            className={cn(
              'text-[14px] font-light tracking-[-0.022em] transition-colors',
              pathname === l.to ? 'text-[#cc9166]' : 'text-[#9194a1] hover:text-[#e2e3e9]',
            )}
          >
            {l.label}
          </Link>
        ))}
      </div>

      <Link
        to="/showdown"
        className="rounded-[2px] bg-white px-5 py-[8px] text-[13px] font-medium text-[#08080a] transition-opacity hover:opacity-90"
      >
        Run demo →
      </Link>
    </nav>
  )
}
