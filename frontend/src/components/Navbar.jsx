import { Link } from 'react-router-dom'

export default function Navbar() {
  return (
    <nav
      className="sticky top-0 z-50 flex h-14 items-center justify-between border-b border-[#2e3038] bg-[#000000]/95 px-8 backdrop-blur-sm"
      style={{ fontFamily: "'Inter', sans-serif" }}
    >
      <Link to="/" className="text-[15px] font-semibold tracking-[-0.01em] text-white">
        Velora
      </Link>

      <div className="flex items-center gap-8">
        <Link to="/" className="text-[13px] font-medium text-[#9194a1] transition-colors hover:text-[#e2e3e9]">
          About
        </Link>
        <Link to="/execution-demo" className="text-[13px] font-medium text-[#9194a1] transition-colors hover:text-[#e2e3e9]">
          Live Demo
        </Link>
      </div>

      <Link
        to="/execution-demo"
        className="rounded-[2px] bg-white px-4 py-[7px] text-[13px] font-medium text-[#08080a] transition-opacity hover:opacity-90"
      >
        Run demo →
      </Link>
    </nav>
  )
}
