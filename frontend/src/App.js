import React, { useEffect, useState } from "react";
import { Routes, Route, Link } from "react-router-dom";
import Dashboard from "./Dashboard";
import Metrics from "./Metrics";
import axios from "axios";
import "./App.css";

function App() {
  const [dataStatus, setDataStatus] = useState({
    metrics_available: false,
    abtest_available: false,
  });

  useEffect(() => {
    axios
      .get("/data_status")
      .then((res) => setDataStatus(res.data))
      .catch((err) => console.error("Data status fetch failed:", err));
  }, []);
  return (
    <div className="app-container">
      <nav className="sidebar">
        <h2>LeadHeatScore</h2>
        <ul>
          <li>
            <Link to="/">Dashboard</Link>
          </li>
          {dataStatus.metrics_available && (
            <li>
              <Link to="/metrics">Metrics & Tests</Link>
            </li>
          )}
        </ul>
      </nav>

      <main className="main-content">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          {dataStatus.metrics_available && (
            <Route path="/metrics" element={<Metrics />} />
          )}
        </Routes>
      </main>
    </div>
  );
}

export default App;
