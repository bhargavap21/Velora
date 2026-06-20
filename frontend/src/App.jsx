import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Home from './pages/Home'
import LegacyHome from './pages/LegacyHome'
import Live from './pages/Live'
import Results from './pages/Results'
import ExecutionLive from './pages/ExecutionLive'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/legacy" element={<LegacyHome />} />
        <Route path="/live/:runId" element={<Live />} />
        <Route path="/results/:runId" element={<Results />} />
        <Route path="/execution-demo" element={<ExecutionLive />} />
      </Routes>
    </BrowserRouter>
  )
}
