import { Link } from 'react-router-dom'

export default function Navbar() {
  return (
    <nav
      className="sticky top-0 z-50 flex h-16 items-center justify-between border-b border-[#2e3038] bg-[#000000]/95 px-12 backdrop-blur-sm"
      style={{ fontFamily: "'Inter', sans-serif" }}
    >
      <Link
        to="/"
        className="text-[22px] text-white"
        style={{ fontFamily: "'Playfair Display', Georgia, serif", fontStyle: 'italic', fontWeight: 400 }}
      >
        Velora
      </Link>

      <div className="flex items-center gap-10">
        <Link to="/" className="text-[14px] font-light tracking-[-0.022em] text-[#9194a1] transition-colors hover:text-[#e2e3e9]">
          About
        </Link>
        <Link to="/execution-demo" className="text-[14px] font-light tracking-[-0.022em] text-[#9194a1] transition-colors hover:text-[#e2e3e9]">
          Live Demo
        </Link>
      </div>

      <Link
        to="/execution-demo"
        className="rounded-[2px] bg-white px-5 py-[8px] text-[13px] font-medium text-[#08080a] transition-opacity hover:opacity-90"
      >
        Run demo →
      </Link>
    </nav>
  )
}
