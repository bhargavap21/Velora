import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Home from './pages/Home'
import ExecutionLive from './pages/ExecutionLive'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/execution-demo" element={<ExecutionLive />} />
      </Routes>
    </BrowserRouter>
  )
}
