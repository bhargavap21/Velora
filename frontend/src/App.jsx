import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Home from './pages/Home'
import Showdown from './pages/Showdown'
import Proof from './pages/Proof'
import ExecutionLive from './pages/ExecutionLive'
import ExecutionSandbox from './pages/ExecutionSandbox'
import Pitch from './pages/Pitch'
import RftLive from './pages/RftLive'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/showdown" element={<Showdown />} />
        <Route path="/proof" element={<Proof />} />
        <Route path="/execution-demo" element={<ExecutionLive />} />
        <Route path="/sandbox" element={<ExecutionSandbox />} />
        <Route path="/pitch" element={<Pitch />} />
        <Route path="/rft" element={<RftLive />} />
      </Routes>
    </BrowserRouter>
  )
}
