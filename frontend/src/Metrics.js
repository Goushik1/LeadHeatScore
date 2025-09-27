import React, { useEffect, useState } from "react";
import axios from "axios";

function Metrics() {
  const [metrics, setMetrics] = useState(null);
  useEffect(() => {
    axios
      .get("http://localhost:5000/metrics")
      .then((res) => setMetrics(res.data));
  }, []);
  const [abTest, setAbTest] = useState(null);
  useEffect(() => {
    axios
      .get("http://localhost:5000/abtest_summary")
      .then((res) => setAbTest(res.data));
  }, []);
  if (!metrics || !abTest) return <p className="loading">Loading metrics...</p>;

  return (
    <div className="metrics-container">
      <h1>Metrics & Tests</h1>

      <div className="card">
        <h3>Model Performance</h3>
        <p>
          <strong>Macro F1:</strong> {metrics.f1_macro.toFixed(2)}
        </p>
        <p>
          <strong>ROC AUC:</strong> {metrics.roc_auc.toFixed(2)}
        </p>
        <p>
          <strong>Brier Score:</strong> {metrics.brier_score.toFixed(3)}
        </p>
        <h4>Confusion Matrix:</h4>
        <pre>{JSON.stringify(metrics.confusion_matrix, null, 2)}</pre>
        <h4>Reliability Plot:</h4>
        <img
          src={`data:image/png;base64,${metrics.reliability_plot}`}
          alt="Reliability"
        />
      </div>

      <div className="card">
        <h3>A/B Test Summary</h3>
        {Object.entries(abTest).map(([key, val]) =>
          Array.isArray(val) ? (
            <div key={key}>
              <h4>{key}:</h4>
              {val.map((lead) => (
                <p key={lead.lead_id}>
                  {lead.lead_id} - Class: {lead.class} - Score: {lead.score}
                </p>
              ))}
            </div>
          ) : (
            <p key={key}>
              <strong>{key}:</strong> {val.toFixed(1)}
            </p>
          )
        )}
      </div>
    </div>
  );
}

export default Metrics;
