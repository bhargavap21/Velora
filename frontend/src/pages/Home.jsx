import { useNavigate } from 'react-router-dom'
import Navbar from '../components/Navbar'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card'

const FACTS = [
  { label: 'Benchmark', value: 'VWAP slippage' },
  { label: 'Agents', value: 'PPO + LLM' },
  { label: 'Built on', value: 'HUD environment' },
]

export default function Home() {
  const navigate = useNavigate()

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Navbar />

      <div className="mx-auto max-w-3xl px-4 py-24 text-center">
        <Badge variant="secondary" className="mb-4">HUD x YC Hackathon</Badge>
        <h1 className="text-4xl font-bold tracking-tight sm:text-5xl">Velora</h1>
        <p className="mt-4 text-lg text-muted-foreground">
          An RL environment for optimal trade execution.
        </p>
        <p className="mx-auto mt-3 max-w-xl text-sm text-muted-foreground">
          Dump a large order at once and you move the price against yourself. Spread it
          out wrong and you eat the day's drift instead. Velora trains and benchmarks
          agents — classical RL and LLM — on filling large orders against the same
          metric real execution desks are graded on: slippage vs. VWAP.
        </p>

        <div className="mt-8 flex items-center justify-center gap-3">
          <Button size="lg" onClick={() => navigate('/sandbox')}>
            Open sandbox →
          </Button>
          <Button size="lg" variant="outline" onClick={() => navigate('/execution-demo')}>
            Watch live demo
          </Button>
        </div>

        <div className="mx-auto mt-16 grid max-w-lg grid-cols-3 gap-4">
          {FACTS.map(f => (
            <Card key={f.label}>
              <CardContent className="pt-6 text-center">
                <p className="text-xs text-muted-foreground">{f.label}</p>
                <p className="mt-1 text-sm font-semibold">{f.value}</p>
              </CardContent>
            </Card>
          ))}
        </div>

        <Card className="mt-10 text-left">
          <CardHeader>
            <CardTitle className="text-base">Who this is for</CardTitle>
            <CardDescription>
              Execution desks, prop trading firms, and asset managers measured on
              slippage vs. VWAP every day — plus quant teams who want a realistic,
              configurable RL sandbox to train and benchmark execution agents before
              risking real capital.
            </CardDescription>
          </CardHeader>
        </Card>
      </div>
    </div>
  )
}
