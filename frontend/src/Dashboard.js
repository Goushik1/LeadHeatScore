import React, { useState } from "react";
import axios from "axios";
import Papa from "papaparse";

export default function LeadDashboard() {
  const [leads, setLeads] = useState([]);
  const [selectedLeads, setSelectedLeads] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleCSVUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setLoading(true);
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      complete: async (results) => {
        const parsedLeads = results.data;
        const scoredLeads = await Promise.all(
          parsedLeads.map(async (lead) => {
            const response = await axios.post("http://localhost:5000/score", {
              lead_json: lead,
            });
            return { ...lead, ...response.data };
          })
        );
        setLeads(scoredLeads);
        setLoading(false);
      },
    });
  };

  const pickTopLeads = () => {
    const classes = ["Hot", "Warm", "Cold"];
    let picked = [];

    classes.forEach((cls) => {
      const filtered = leads
        .filter((l) => l.class === cls)
        .sort((a, b) => b.prob - a.prob)
        .slice(0, 3);
      picked = [...picked, ...filtered];
    });

    setSelectedLeads(picked);
  };

  const generateNextAction = async () => {
    setLoading(true);
    const recs = await Promise.all(
      selectedLeads.map(async (lead) => {
        const response = await axios.post("http://localhost:5000/recommend", {
          lead_json: lead,
        });
        // Include the lead class here
        return { lead_id: lead.lead_id, class: lead.class, ...response.data };
      })
    );
    setRecommendations(recs);
    setLoading(false);
  };

  return (
    <div className="container">
      <h1>Lead Dashboard</h1>
      <input type="file" accept=".csv" onChange={handleCSVUpload} />

      {loading && <p>Processing... Please wait.</p>}

      <div style={{ marginTop: 20 }}>
        <button onClick={pickTopLeads}>Pick Top 3 Leads per Class</button>
        <button onClick={generateNextAction} style={{ marginLeft: 10 }}>
          Generate Next Action
        </button>
      </div>

      <div className="top-leads-container">
        {selectedLeads.map((l) => (
          <div
            key={l.lead_id}
            className={`top-lead-card ${l.class.toLowerCase()}`} // hot/warm/cold
          >
            <h4>{l.lead_id}</h4>
            <p>
              <strong>Class:</strong> {l.class}
            </p>
            <p>
              <strong>Probability:</strong> {l.prob.toFixed(3)}
            </p>
          </div>
        ))}
      </div>

      {recommendations.map((r) => (
        <div
          key={r.lead_id}
          className={`recommendation-card ${r.class?.toLowerCase()}`}
        >
          <p>
            <strong>Lead:</strong> {r.lead_id}
          </p>
          <p>
            <strong>Channel:</strong> {r.channel}
          </p>
          <p>
            <strong>Message:</strong> {r.message}
          </p>
          <p>
            <strong>Rationale:</strong> {r.rationale}
          </p>
        </div>
      ))}
    </div>
  );
}
