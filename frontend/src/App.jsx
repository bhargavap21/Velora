import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Home from './pages/Home'
import Live from './pages/Live'
import Results from './pages/Results'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/live/:runId" element={<Live />} />
        <Route path="/results/:runId" element={<Results />} />
      </Routes>
    </BrowserRouter>
  )
}
