import { useSearchParams } from 'react-router-dom'
import Navbar from '../components/Navbar'
import PolicyReplayPanel from '../components/PolicyReplayPanel'

export default function ReplayPage() {
  const [params] = useSearchParams()
  const runId = params.get('run_id') ?? undefined

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Navbar />
      <PolicyReplayPanel runId={runId} />
    </div>
  )
}
