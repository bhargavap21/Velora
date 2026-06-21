import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Home from './pages/Home'
import Showdown from './pages/Showdown'
import Proof from './pages/Proof'
import ExecutionLive from './pages/ExecutionLive'
import ExecutionSandbox from './pages/ExecutionSandbox'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/showdown" element={<Showdown />} />
        <Route path="/proof" element={<Proof />} />
        <Route path="/execution-demo" element={<ExecutionLive />} />
        <Route path="/sandbox" element={<ExecutionSandbox />} />
      </Routes>
    </BrowserRouter>
  )
}
