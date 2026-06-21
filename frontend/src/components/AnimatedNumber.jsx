import { useEffect, useRef, useState } from 'react'

// Smoothly tweens a displayed number toward `value` whenever it changes. Used for the
// live scoreboard / headline stats so figures count up rather than snapping.
export default function AnimatedNumber({
  value,
  duration = 600,
  format = (n) => n.toFixed(1),
  className,
}) {
  const [display, setDisplay] = useState(value ?? 0)
  const fromRef = useRef(value ?? 0)
  const rafRef = useRef(null)
  const startRef = useRef(0)

  useEffect(() => {
    if (value == null || Number.isNaN(value)) return
    const from = fromRef.current
    const to = value
    if (from === to) return
    startRef.current = performance.now()

    const tick = (now) => {
      const t = Math.min(1, (now - startRef.current) / duration)
      // ease-out cubic
      const eased = 1 - Math.pow(1 - t, 3)
      const current = from + (to - from) * eased
      setDisplay(current)
      if (t < 1) {
        rafRef.current = requestAnimationFrame(tick)
      } else {
        fromRef.current = to
      }
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(rafRef.current)
  }, [value, duration])

  return <span className={className}>{format(display)}</span>
}
