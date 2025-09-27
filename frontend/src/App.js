import { Routes, Route, Link } from "react-router-dom";
import Dashboard from "./Dashboard";
import Metrics from "./Metrics";
import "./App.css";

function App() {
  return (
    <div className="app-container">
      <nav className="sidebar">
        <h2>LeadHeatScore</h2>
        <ul>
          <li>
            <Link to="/">Dashboard</Link>
          </li>
          <li>
            <Link to="/metrics">Metrics & Tests</Link>
          </li>
        </ul>
      </nav>

      <main className="main-content">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/metrics" element={<Metrics />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
