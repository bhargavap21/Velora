import { Link } from 'react-router-dom'

export default function Navbar() {
  return (
    <nav className="flex h-14 items-center justify-between border-b border-border bg-card px-6">
      <Link to="/" className="text-lg font-bold tracking-tight">Velora</Link>
      <div className="flex items-center gap-4 text-sm text-muted-foreground">
        <Link to="/" className="hover:text-foreground">Home</Link>
        <Link to="/execution-demo" className="hover:text-foreground">Live Demo</Link>
        <Link to="/sandbox" className="hover:text-foreground">Sandbox</Link>
      </div>
    </nav>
  )
}
