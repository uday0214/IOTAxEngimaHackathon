"""Visualization utilities for telemetry trajectories, model diagnostics, and supervisor reports."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def generate_interactive_dashboard(
    summary_path: Path = Path("artifacts/telemetry/full_trajectory_summary.json"),
    output_html: Path = Path("artifacts/telemetry/trajectory_dashboard.html"),
) -> Path:
    """Generates a standalone, beautiful HTML interactive dashboard for training telemetry."""
    summary_path = Path(summary_path)
    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    if not summary_path.exists():
        logger.warning("Summary file %s does not exist. Cannot generate dashboard.", summary_path)
        return output_html

    with open(summary_path, encoding="utf-8") as f:
        data = json.load(f)

    time_series = data.get("trajectory_time_series", [])
    champ = data.get("champion_metrics", {})
    base = data.get("baseline_metrics", {})
    net = data.get("net_improvements", {})
    curves = data.get("curves_data", {})

    runs_json = json.dumps(time_series)
    curves_json = json.dumps(curves)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Utkrishta 2026 - Autonomous AI Data Scientist Trajectory Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: #0f172a;
      color: #f8fafc;
      margin: 0;
      padding: 24px;
    }}
    .container {{
      max-width: 1200px;
      margin: 0 auto;
    }}
    h1 {{
      font-size: 28px;
      margin-bottom: 8px;
      color: #38bdf8;
    }}
    .subtitle {{
      color: #94a3b8;
      margin-bottom: 24px;
    }}
    .grid-cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .card {{
      background: #1e293b;
      padding: 20px;
      border-radius: 12px;
      border: 1px solid #334155;
    }}
    .card-title {{
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: #94a3b8;
      margin-bottom: 8px;
    }}
    .card-value {{
      font-size: 28px;
      font-weight: 700;
      color: #38bdf8;
    }}
    .card-delta {{
      font-size: 13px;
      margin-top: 6px;
      color: #4ade80;
    }}
    .chart-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(540px, 1fr));
      gap: 20px;
      margin-bottom: 24px;
    }}
    .chart-box {{
      background: #1e293b;
      padding: 20px;
      border-radius: 12px;
      border: 1px solid #334155;
    }}
    .chart-box h3 {{
      margin-top: 0;
      font-size: 16px;
      color: #e2e8f0;
      margin-bottom: 16px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
      font-size: 13px;
    }}
    th, td {{
      padding: 10px 12px;
      text-align: left;
      border-bottom: 1px solid #334155;
    }}
    th {{
      background: #0f172a;
      color: #94a3b8;
    }}
    tr:hover {{
      background: #334155;
    }}
    .badge {{
      background: #0369a1;
      padding: 3px 8px;
      border-radius: 6px;
      font-weight: 600;
      font-size: 11px;
    }}
  </style>
</head>
<body>
<div class="container">
  <h1>Autonomous AI Data Scientist: Telemetry & Trajectory Dashboard</h1>
  <div class="subtitle">Utkrishta 2026 Challenge 1 | 15-Experiment Autonomous Supervisor Optimization Loop</div>

  <div class="grid-cards">
    <div class="card">
      <div class="card-title">Champion ROC-AUC (OOF)</div>
      <div class="card-value">{champ.get('oof_roc_auc', 0.0):.5f}</div>
      <div class="card-delta">+{net.get('delta_roc_auc', 0.0):.5f} ({net.get('pct_gain_roc_auc', 0.0):+.2f}%)</div>
    </div>
    <div class="card">
      <div class="card-title">Champion PR-AUC (OOF)</div>
      <div class="card-value">{champ.get('oof_pr_auc', 0.0):.5f}</div>
      <div class="card-delta">+{net.get('delta_pr_auc', 0.0):.5f} ({net.get('pct_gain_pr_auc', 0.0):+.2f}%)</div>
    </div>
    <div class="card">
      <div class="card-title">Overfitting Index (Omega)</div>
      <div class="card-value">{champ.get('overfitting_index', 0.0):.2f}%</div>
      <div class="card-delta" style="color: #60a5fa;">Compressed from {base.get('overfitting_index', 0.0):.2f}%</div>
    </div>
    <div class="card">
      <div class="card-title">Calibration Error (ECE)</div>
      <div class="card-value">{champ.get('ece_score', 0.0):.4f}</div>
      <div class="card-delta" style="color: #4ade80;">Well-Calibrated (Youden J* = {champ.get('optimal_youden_threshold', 0.38):.3f})</div>
    </div>
  </div>

  <div class="chart-grid">
    <div class="chart-box">
      <h3>Metric Progression Across 15 Runs</h3>
      <canvas id="trajectoryChart" height="200"></canvas>
    </div>
    <div class="chart-box">
      <h3>Overfitting Index & Loss Gap</h3>
      <canvas id="overfittingChart" height="200"></canvas>
    </div>
  </div>

  <div class="chart-box" style="margin-bottom: 24px;">
    <h3>Full Experiment Runs Progression Table</h3>
    <div style="overflow-x: auto;">
      <table>
        <thead>
          <tr>
            <th>Run</th>
            <th>Depth</th>
            <th>L2 Reg</th>
            <th>LR</th>
            <th>Iter</th>
            <th>Class Weight</th>
            <th>ROC-AUC</th>
            <th>PR-AUC</th>
            <th>Log Loss</th>
            <th>Omega (%)</th>
            <th>ECE</th>
          </tr>
        </thead>
        <tbody id="tableBody"></tbody>
      </table>
    </div>
  </div>
</div>

<script>
  const runs = {runs_json};
  const curves = {curves_json};

  // Populate Table
  const tbody = document.getElementById('tableBody');
  runs.forEach(r => {{
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><b>#${{r.experiment_id}}</b></td>
      <td>${{r.depth}}</td>
      <td>${{r.l2_leaf_reg.toFixed(1)}}</td>
      <td>${{r.learning_rate.toFixed(3)}}</td>
      <td>${{r.iterations}}</td>
      <td><span class="badge">${{r.class_weight_mode}}</span></td>
      <td style="color:#38bdf8; font-weight:600;">${{r.oof_roc_auc.toFixed(5)}}</td>
      <td>${{r.oof_pr_auc.toFixed(5)}}</td>
      <td>${{r.oof_log_loss.toFixed(4)}}</td>
      <td>${{r.overfitting_index.toFixed(2)}}%</td>
      <td>${{r.ece_score.toFixed(4)}}</td>
    `;
    tbody.appendChild(tr);
  }});

  // Trajectory Chart
  const ctxTraj = document.getElementById('trajectoryChart').getContext('2d');
  new Chart(ctxTraj, {{
    type: 'line',
    data: {{
      labels: runs.map(r => 'Run ' + r.experiment_id),
      datasets: [
        {{
          label: 'OOF ROC-AUC',
          data: runs.map(r => r.oof_roc_auc),
          borderColor: '#38bdf8',
          backgroundColor: 'rgba(56, 189, 248, 0.1)',
          tension: 0.3,
          fill: true
        }},
        {{
          label: 'OOF PR-AUC',
          data: runs.map(r => r.oof_pr_auc),
          borderColor: '#4ade80',
          backgroundColor: 'transparent',
          tension: 0.3,
          borderDash: [5, 5]
        }}
      ]
    }},
    options: {{
      responsive: true,
      plugins: {{ legend: {{ labels: {{ color: '#cbd5e1' }} }} }},
      scales: {{
        x: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }} }},
        y: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }} }}
      }}
    }}
  }});

  // Overfitting Chart
  const ctxOver = document.getElementById('overfittingChart').getContext('2d');
  new Chart(ctxOver, {{
    type: 'bar',
    data: {{
      labels: runs.map(r => 'Run ' + r.experiment_id),
      datasets: [
        {{
          label: 'Overfitting Index (%)',
          data: runs.map(r => r.overfitting_index),
          backgroundColor: runs.map(r => r.overfitting_index > 15 ? '#f87171' : (r.overfitting_index < 3 ? '#fbbf24' : '#38bdf8')),
          borderRadius: 6
        }}
      ]
    }},
    options: {{
      responsive: true,
      plugins: {{ legend: {{ labels: {{ color: '#cbd5e1' }} }} }},
      scales: {{
        x: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }} }},
        y: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }} }}
      }}
    }}
  }});
</script>
</body>
</html>
"""

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info("Generated standalone interactive HTML dashboard at %s", output_html)
    return output_html
