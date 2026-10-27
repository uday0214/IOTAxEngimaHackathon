import json
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.generate_visualization_website import load_artifacts

def main():
    runs, holdout, report_md, submission_rows, supervisor_json = load_artifacts()

    runs_json = json.dumps(runs)
    holdout_json = json.dumps(holdout)
    supervisor_json_str = json.dumps(supervisor_json)

    # Tournament leaderboard
    tourney_leaderboard = supervisor_json.get("tournament_leaderboard", runs[:5])
    champ_id = holdout.get("champion_experiment_id", 12)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Linear • Customer Retention Intelligence</title>
  
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  
  <!-- Three.js 3D Engine & Chart.js -->
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

  <style>
    :root {{
      /* Linear Dark High-End Architectural Palette */
      --bg-canvas: #08090A;
      --bg-surface: rgba(15, 16, 18, 0.82);
      --bg-surface-elevated: rgba(22, 24, 28, 0.88);
      --bg-surface-hover: rgba(28, 31, 36, 0.92);
      
      --border-subtle: rgba(255, 255, 255, 0.08);
      --border-medium: rgba(255, 255, 255, 0.14);
      --border-focus: rgba(255, 255, 255, 0.25);
      
      --text-high: #F7F8F8;
      --text-medium: #8A8F98;
      --text-dim: #545861;
      
      --accent-purple: #5E6AD2;
      --accent-purple-dim: rgba(94, 106, 210, 0.15);
      --accent-emerald: #4EBE9E;
      --accent-emerald-dim: rgba(78, 190, 158, 0.14);
      --accent-cyan: #38BDF8;
      --accent-cyan-dim: rgba(56, 189, 248, 0.14);
      --accent-amber: #F59E0B;
      --accent-rose: #F43F5E;

      --font-sans: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Inter', sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }}

    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    body {{
      font-family: var(--font-sans);
      background-color: var(--bg-canvas);
      color: var(--text-high);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      -webkit-font-smoothing: antialiased;
      overflow-x: hidden;
      position: relative;
    }}

    /* 3D INTERACTIVE CANVAS BACKGROUND */
    #webgl-canvas {{
      position: fixed;
      top: 0;
      left: 0;
      width: 100vw;
      height: 100vh;
      z-index: 0;
      pointer-events: none;
    }}

    /* INTERACTIVE MOUSE LIGHT GLOW */
    .cursor-glow {{
      position: fixed;
      width: 580px;
      height: 580px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(94, 106, 210, 0.08) 0%, rgba(94, 106, 210, 0.02) 45%, transparent 70%);
      transform: translate(-50%, -50%);
      pointer-events: none;
      z-index: 0;
      will-change: left, top;
      transition: opacity 0.3s ease;
    }}

    /* AMBIENT RADIAL TOP GLOW */
    .ambient-glow {{
      position: fixed;
      top: 0;
      left: 50%;
      transform: translateX(-50%);
      width: 1100px;
      height: 480px;
      background: radial-gradient(ellipse at 50% 0%, rgba(94, 106, 210, 0.12) 0%, rgba(8, 9, 10, 0) 72%);
      pointer-events: none;
      z-index: 0;
    }}

    .container {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 0 52px;
      position: relative;
      z-index: 10;
      width: 100%;
    }}

    /* TOP MINIMALIST NAV */
    .top-nav {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      height: 72px;
      border-bottom: 1px solid var(--border-subtle);
      background: rgba(8, 9, 10, 0.7);
      backdrop-filter: blur(14px);
      -webkit-backdrop-filter: blur(14px);
      position: sticky;
      top: 0;
      z-index: 100;
      padding: 0 52px;
    }}

    .nav-brand {{
      display: flex;
      align-items: center;
      gap: 12px;
      font-size: 14px;
      font-weight: 600;
      letter-spacing: -0.01em;
      color: var(--text-high);
      cursor: pointer;
    }}

    .nav-logo-mark {{
      width: 18px;
      height: 18px;
      border-radius: 4px;
      background: linear-gradient(135deg, #F7F8F8 0%, #8A8F98 100%);
      display: inline-block;
    }}

    .nav-subsections {{
      display: flex;
      align-items: center;
      gap: 4px;
      background: rgba(255, 255, 255, 0.03);
      padding: 3px;
      border-radius: 8px;
      border: 1px solid var(--border-subtle);
    }}

    .sub-tab {{
      background: transparent;
      border: none;
      outline: none;
      padding: 6px 14px;
      font-family: var(--font-sans);
      font-size: 12px;
      font-weight: 500;
      color: var(--text-medium);
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.15s ease;
    }}

    .sub-tab:hover {{
      color: var(--text-high);
    }}

    .sub-tab.active {{
      background: var(--bg-surface-elevated);
      color: var(--text-high);
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.4);
    }}

    .nav-actions {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}

    /* ACTION BUTTONS */
    .linear-btn {{
      background: var(--bg-surface-elevated);
      border: 1px solid var(--border-medium);
      color: var(--text-high);
      font-family: var(--font-sans);
      font-size: 12.5px;
      font-weight: 500;
      padding: 7px 16px;
      border-radius: 6px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      transition: all 0.15s ease;
    }}

    .linear-btn:hover {{
      background: var(--bg-surface-hover);
      border-color: var(--border-focus);
    }}

    .linear-btn.primary {{
      background: #F7F8F8;
      color: #08090A;
      border-color: #F7F8F8;
      font-weight: 600;
    }}

    .linear-btn.primary:hover {{
      background: #E1E3E5;
    }}

    /* MAIN SECTIONS CONTROLLER */
    main {{
      flex: 1;
      padding: 48px 0 96px;
    }}

    .view-subsection {{
      display: none;
      animation: fadeIn 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    }}

    .view-subsection.active {{
      display: block;
    }}

    @keyframes fadeIn {{
      from {{ opacity: 0; transform: translateY(6px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}

    /* HERO EDITORIAL */
    .hero-container {{
      padding: 40px 0 64px;
      max-width: 860px;
    }}

    .hero-editorial {{
      font-size: 48px;
      line-height: 1.12;
      font-weight: 600;
      letter-spacing: -0.03em;
      color: var(--text-high);
      margin-bottom: 24px;
    }}

    .hero-editorial .editorial-muted {{
      display: block;
      color: var(--text-dim);
      font-weight: 400;
      margin-top: 6px;
    }}

    .hero-actions-row {{
      display: flex;
      align-items: center;
      gap: 14px;
      margin-top: 32px;
    }}

    /* THREE PILLARS (FIG 0.1, FIG 0.2, FIG 0.3) */
    .pillars-grid {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 24px;
      margin-bottom: 64px;
    }}

    .pillar-card {{
      background: rgba(15, 16, 18, 0.4);
      border: 1px solid transparent;
      border-radius: 8px;
      padding: 24px 20px;
      position: relative;
      cursor: pointer;
      transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    }}

    .pillar-card:hover {{
      background: var(--bg-surface);
      border-color: var(--border-medium);
      transform: translateY(-2px);
    }}

    .pillar-fig-tag {{
      font-family: var(--font-mono);
      font-size: 10.5px;
      font-weight: 500;
      color: var(--text-dim);
      margin-bottom: 20px;
      letter-spacing: 0.05em;
    }}

    .pillar-visual {{
      height: 120px;
      width: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
      margin-bottom: 24px;
    }}

    .wireframe-svg {{
      width: 120px;
      height: 100px;
      stroke: rgba(255, 255, 255, 0.7);
      stroke-width: 1.1;
      fill: none;
      stroke-dasharray: 2 1;
      transition: all 0.3s ease;
    }}

    .pillar-card:hover .wireframe-svg {{
      stroke: var(--accent-purple);
      stroke-dasharray: none;
    }}

    .pillar-title {{
      font-size: 14px;
      font-weight: 600;
      color: var(--text-high);
      margin-bottom: 8px;
    }}

    .pillar-desc {{
      font-size: 12.5px;
      line-height: 1.5;
      color: var(--text-medium);
    }}

    /* HIGH-DENSITY METRIC STRIP */
    .stat-strip {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      background: var(--bg-surface);
      overflow: hidden;
      margin-bottom: 48px;
    }}

    .stat-cell {{
      padding: 24px 28px;
      border-right: 1px solid var(--border-subtle);
      position: relative;
    }}

    .stat-cell:last-child {{
      border-right: none;
    }}

    .stat-label {{
      font-size: 11.5px;
      font-weight: 500;
      color: var(--text-dim);
      margin-bottom: 8px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}

    .stat-num {{
      font-family: var(--font-mono);
      font-size: 26px;
      font-weight: 600;
      color: var(--text-high);
      letter-spacing: -0.02em;
    }}

    .stat-sub {{
      font-size: 11.5px;
      color: var(--text-medium);
      margin-top: 4px;
    }}

    /* SUBSECTION HEADINGS */
    .section-head {{
      margin-bottom: 32px;
    }}

    .section-tag {{
      font-family: var(--font-mono);
      font-size: 11px;
      color: var(--accent-purple);
      margin-bottom: 6px;
      letter-spacing: 0.06em;
    }}

    .section-title {{
      font-size: 24px;
      font-weight: 600;
      letter-spacing: -0.02em;
      color: var(--text-high);
    }}

    .section-sub {{
      font-size: 13.5px;
      color: var(--text-medium);
      margin-top: 6px;
    }}

    /* CHARTS GRIDS & CARDS */
    .chart-filter-bar {{
      display: flex;
      gap: 8px;
      margin-bottom: 24px;
      flex-wrap: wrap;
    }}

    .filter-btn {{
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      color: var(--text-medium);
      padding: 7px 16px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.15s ease;
    }}

    .filter-btn:hover {{
      border-color: var(--border-medium);
      color: var(--text-high);
    }}

    .filter-btn.active {{
      background: var(--accent-purple-dim);
      border-color: var(--accent-purple);
      color: var(--text-high);
      font-weight: 600;
    }}

    .chart-group-title {{
      font-size: 15px;
      font-weight: 600;
      color: var(--text-high);
      margin: 36px 0 16px 0;
      display: flex;
      align-items: center;
      gap: 10px;
    }}

    .chart-group-title .group-tag {{
      font-family: var(--font-mono);
      font-size: 10.5px;
      color: var(--accent-emerald);
      background: var(--accent-emerald-dim);
      padding: 2px 7px;
      border-radius: 4px;
      letter-spacing: 0.04em;
    }}

    .charts-grid-2 {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 16px;
      margin-bottom: 24px;
    }}

    .charts-grid-3 {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 16px;
      margin-bottom: 24px;
    }}

    .charts-duo {{
      display: grid;
      grid-template-columns: 1.4fr 1fr;
      gap: 16px;
      margin-bottom: 24px;
    }}

    .chart-card {{
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 20px 22px;
      position: relative;
    }}

    .chart-card-head {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
    }}

    .chart-heading {{
      font-size: 13.5px;
      font-weight: 600;
      color: var(--text-high);
    }}

    .chart-badge {{
      font-family: var(--font-mono);
      font-size: 11px;
      color: var(--text-medium);
      background: rgba(255, 255, 255, 0.04);
      padding: 3px 8px;
      border-radius: 4px;
      border: 1px solid var(--border-subtle);
    }}

    .chart-canvas-wrapper {{
      height: 250px;
      position: relative;
      width: 100%;
    }}

    .chart-canvas-wrapper-sm {{
      height: 220px;
      position: relative;
      width: 100%;
    }}

    /* TOURNAMENT CARDS */
    .tourney-cards-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 14px;
      margin-bottom: 24px;
    }}

    .tourney-card {{
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 18px;
      position: relative;
      transition: all 0.2s ease;
    }}

    .tourney-card:hover {{
      border-color: var(--border-medium);
      transform: translateY(-2px);
    }}

    .tourney-card.champ {{
      border-color: rgba(78, 190, 158, 0.4);
      background: linear-gradient(180deg, rgba(78, 190, 158, 0.08) 0%, rgba(15, 16, 18, 0.85) 100%);
    }}

    .tourney-rank {{
      font-size: 10px;
      font-family: var(--font-mono);
      color: var(--text-dim);
      margin-bottom: 4px;
    }}

    .tourney-name {{
      font-size: 15px;
      font-weight: 700;
      color: var(--text-high);
      margin-bottom: 12px;
    }}

    .tourney-stat-row {{
      display: flex;
      justify-content: space-between;
      font-size: 12px;
      margin-bottom: 6px;
      color: var(--text-medium);
    }}

    .tourney-stat-val {{
      font-family: var(--font-mono);
      font-weight: 600;
      color: var(--text-high);
    }}

    /* STEPPER BAR */
    .stepper-bar {{
      display: flex;
      gap: 8px;
      overflow-x: auto;
      padding-bottom: 8px;
      margin-bottom: 20px;
    }}

    .step-btn {{
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      color: var(--text-medium);
      padding: 7px 12px;
      border-radius: 6px;
      font-family: var(--font-mono);
      font-size: 11px;
      cursor: pointer;
      white-space: nowrap;
      transition: all 0.2s ease;
    }}

    .step-btn:hover {{
      border-color: var(--border-medium);
      color: var(--text-high);
    }}

    .step-btn.active {{
      background: var(--accent-purple-dim);
      border-color: var(--accent-purple);
      color: var(--text-high);
      font-weight: 600;
    }}

    .step-btn.champ-step {{
      border-color: rgba(78, 190, 158, 0.4);
    }}

    .step-btn.champ-step.active {{
      background: var(--accent-emerald-dim);
      border-color: var(--accent-emerald);
    }}

    .exp-meta-card {{
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 20px;
      margin-bottom: 20px;
    }}

    .pill-tag {{
      font-family: var(--font-mono);
      font-size: 11px;
      padding: 3px 8px;
      border-radius: 4px;
      background: rgba(255, 255, 255, 0.05);
      color: var(--text-medium);
      border: 1px solid var(--border-subtle);
    }}

    .cm-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-top: 16px;
    }}

    .cm-card {{
      background: var(--bg-surface-elevated);
      border: 1px solid var(--border-subtle);
      border-radius: 6px;
      padding: 14px;
      font-family: var(--font-mono);
      font-size: 12px;
    }}

    .cm-title {{
      font-size: 12px;
      font-weight: 600;
      color: var(--text-high);
      margin-bottom: 8px;
      font-family: var(--font-sans);
    }}

    /* INSIGHTS GRID */
    .insights-grid {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 16px;
      margin-top: 24px;
    }}

    .insight-card {{
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 20px;
    }}

    .insight-title {{
      font-size: 13px;
      font-weight: 600;
      color: var(--text-high);
      margin-bottom: 6px;
    }}

    .insight-body {{
      font-size: 12px;
      line-height: 1.5;
      color: var(--text-medium);
    }}

    /* PARAMETERS COMPARISON */
    .params-comparison-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
    }}

    .param-box {{
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 24px;
    }}

    .param-box.champion {{
      border-color: rgba(78, 190, 158, 0.4);
      background: linear-gradient(180deg, rgba(78, 190, 158, 0.05) 0%, rgba(15, 16, 18, 0.82) 100%);
    }}

    .param-box-head {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--border-subtle);
    }}

    .param-pill {{
      font-family: var(--font-mono);
      font-size: 10px;
      font-weight: 600;
      padding: 3px 8px;
      border-radius: 4px;
    }}

    .param-pill.champ {{
      background: var(--accent-emerald-dim);
      color: var(--accent-emerald);
      border: 1px solid rgba(78, 190, 158, 0.3);
    }}

    .param-pill.trial {{
      background: rgba(255, 255, 255, 0.05);
      color: var(--text-medium);
      border: 1px solid var(--border-subtle);
    }}

    .param-spec-row {{
      display: flex;
      justify-content: space-between;
      padding: 10px 0;
      border-bottom: 1px solid rgba(255, 255, 255, 0.03);
      font-size: 12.5px;
    }}

    .param-spec-row:last-child {{
      border-bottom: none;
    }}

    .spec-key {{
      color: var(--text-medium);
    }}

    .spec-val {{
      font-family: var(--font-mono);
      color: var(--text-high);
      font-weight: 500;
    }}

    /* HOLDOUT EVALUATION */
    .holdout-panel {{
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 36px;
    }}

    .policy-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12.5px;
      margin-top: 24px;
    }}

    .policy-table th {{
      text-align: left;
      font-weight: 500;
      color: var(--text-dim);
      padding: 12px 16px;
      border-bottom: 1px solid var(--border-medium);
      font-size: 11.5px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}

    .policy-table td {{
      padding: 14px 16px;
      border-bottom: 1px solid var(--border-subtle);
      color: var(--text-medium);
    }}

    .policy-table tr:hover td {{
      background: rgba(255, 255, 255, 0.02);
      color: var(--text-high);
    }}

    .policy-table tr.highlight-policy td {{
      background: rgba(78, 190, 158, 0.06);
      color: var(--text-high);
      font-weight: 500;
    }}

    /* LEDGER TABLE */
    .ledger-table-wrap {{
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      overflow-x: auto;
    }}

    .ledger-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      font-family: var(--font-mono);
    }}

    .ledger-table th {{
      text-align: left;
      padding: 14px 18px;
      color: var(--text-dim);
      font-weight: 500;
      border-bottom: 1px solid var(--border-medium);
      text-transform: uppercase;
      letter-spacing: 0.04em;
      font-size: 11px;
    }}

    .ledger-table td {{
      padding: 12px 18px;
      border-bottom: 1px solid var(--border-subtle);
      color: var(--text-medium);
      cursor: pointer;
    }}

    .ledger-table tr:hover td {{
      background: rgba(255, 255, 255, 0.03);
      color: var(--text-high);
    }}

    .ledger-table tr.champ-row td {{
      background: rgba(78, 190, 158, 0.05);
      color: var(--text-high);
      font-weight: 600;
    }}


    /* FOOTER */
    footer {{
      border-top: 1px solid var(--border-subtle);
      padding: 48px 0 64px;
      text-align: center;
      font-size: 12px;
      color: var(--text-dim);
    }}

    @media (max-width: 900px) {{
      .hero-editorial {{ font-size: 36px; }}
      .pillars-grid {{ grid-template-columns: 1fr; gap: 40px; }}
      .stat-strip {{ grid-template-columns: 1fr 1fr; }}
      .charts-grid-2 {{ grid-template-columns: 1fr; }}
      .charts-grid-3 {{ grid-template-columns: 1fr; }}
      .charts-duo {{ grid-template-columns: 1fr; }}
      .params-comparison-grid {{ grid-template-columns: 1fr; }}
      .insights-grid {{ grid-template-columns: 1fr; }}
      .container {{ padding: 0 24px; }}
      .top-nav {{ flex-wrap: wrap; height: auto; padding: 16px 0; gap: 14px; }}
      .nav-subsections {{ overflow-x: auto; width: 100%; }}
    }}
  </style>
</head>
<body>

  <!-- INTERACTIVE 3D WEBGL CANVAS -->
  <canvas id="webgl-canvas"></canvas>

  <!-- MOUSE TRACKING GLOW LIGHT -->
  <div class="cursor-glow" id="cursorGlow"></div>

  <!-- AMBIENT TOP LIGHT -->
  <div class="ambient-glow"></div>

  <!-- NAVIGATION -->
  <header class="top-nav">
    <div class="nav-brand" onclick="switchView('overview')">
      <span class="nav-logo-mark"></span>
      <span>Linear Retention</span>
    </div>

    <!-- 6 SUBSECTIONS NAVIGATION TABS -->
    <nav class="nav-subsections">
      <button class="sub-tab active" id="tab-overview" onclick="switchView('overview')">Overview</button>
      <button class="sub-tab" id="tab-trajectory" onclick="switchView('trajectory')">Charts</button>
      <button class="sub-tab" id="tab-params" onclick="switchView('params')">Parameters</button>
      <button class="sub-tab" id="tab-holdout" onclick="switchView('holdout')">Holdout Test</button>
      <button class="sub-tab" id="tab-ledger" onclick="switchView('ledger')">Ledger</button>
    </nav>

    <div class="nav-actions">
      <button class="linear-btn" onclick="stepRun()">
        <span id="stepActionText">Step Run (1/15)</span>
      </button>
    </div>
  </header>

  <div class="container">
    <main>

      <!-- ========================================== -->
      <!-- SUBSECTION 1: OVERVIEW & PILLARS           -->
      <!-- ========================================== -->
      <section class="view-subsection active" id="sub-overview">
        
        <!-- MINIMAL HERO HEADLINE -->
        <div class="hero-container">
          <h1 class="hero-editorial">
            A new standard for retention.
            <span class="editorial-muted">Autonomous model exploration across classifiers.</span>
          </h1>
          <div class="hero-actions-row">
            <button class="linear-btn primary" onclick="switchView('trajectory')">Explore Visualizations →</button>
            <button class="linear-btn" onclick="switchView('holdout')">Holdout Results</button>
          </div>
        </div>

        <!-- THREE PILLARS (FIG 0.1, FIG 0.2, FIG 0.3) -->
        <div class="pillars-grid">
          
          <!-- FIG 0.1 -->
          <div class="pillar-card" onclick="switchView('trajectory')">
            <div class="pillar-fig-tag">FIG 0.1</div>
            <div class="pillar-visual">
              <svg class="wireframe-svg" viewBox="0 0 160 120">
                <path d="M80 15 L135 42 L80 69 L25 42 Z"/>
                <ellipse cx="80" cy="42" rx="20" ry="10"/>
                <path d="M25 42 L25 54 L80 81 L135 54 L135 42"/>
                <path d="M25 54 L25 66 L80 93 L135 66 L135 54"/>
                <path d="M25 66 L25 78 L80 105 L135 78 L135 66"/>
                <line x1="80" y1="69" x2="80" y2="105"/>
              </svg>
            </div>
            <div class="pillar-title">Multi-Classifier Tournament</div>
            <div class="pillar-desc">
              CatBoost, LightGBM, XGBoost, Random Forest, Logistic Regression.
            </div>
          </div>

          <!-- FIG 0.2 -->
          <div class="pillar-card" onclick="switchView('ledger')">
            <div class="pillar-fig-tag">FIG 0.2</div>
            <div class="pillar-visual">
              <svg class="wireframe-svg" viewBox="0 0 160 120">
                <path d="M80 20 L105 32 L80 44 L55 32 Z"/>
                <path d="M55 32 L55 58 L80 70 L105 58 L105 32"/>
                <line x1="80" y1="44" x2="80" y2="70"/>
                <path d="M108 42 L133 54 L108 66 L83 54 Z"/>
                <path d="M83 54 L83 80 L108 92 L133 80 L133 54"/>
                <line x1="108" y1="66" x2="108" y2="92"/>
                <path d="M52 42 L77 54 L52 66 L27 54 Z"/>
                <path d="M27 54 L27 80 L52 92 L77 80 L77 54"/>
                <line x1="52" y1="66" x2="52" y2="92"/>
              </svg>
            </div>
            <div class="pillar-title">15-Run Budget</div>
            <div class="pillar-desc">
              Calibrated search with early stopping.
            </div>
          </div>

          <!-- FIG 0.3 -->
          <div class="pillar-card" onclick="switchView('holdout')">
            <div class="pillar-fig-tag">FIG 0.3</div>
            <div class="pillar-visual">
              <svg class="wireframe-svg" viewBox="0 0 160 120">
                <path d="M30 95 L60 80 L60 30 L30 45 Z"/>
                <path d="M45 90 L75 75 L75 25 L45 40 Z"/>
                <path d="M60 85 L90 70 L90 20 L60 35 Z"/>
                <path d="M75 80 L105 65 L105 15 L75 30 Z"/>
                <path d="M90 75 L120 60 L120 10 L90 25 Z"/>
                <line x1="30" y1="95" x2="120" y2="60"/>
              </svg>
            </div>
            <div class="pillar-title">Holdout Verification</div>
            <div class="pillar-desc">
              1,409 accounts evaluated at 0.490 cutoff.
            </div>
          </div>

        </div>

        <!-- STAT STRIP -->
        <div class="stat-strip">
          <div class="stat-cell">
            <div class="stat-label">Top Classifier</div>
            <div class="stat-num" id="statBestModel">CatBoost</div>
            <div class="stat-sub" id="statBestRun">Run #12 (Champion)</div>
          </div>
          <div class="stat-cell">
            <div class="stat-label">Validation ROC</div>
            <div class="stat-num" id="statTopVal">0.8506</div>
            <div class="stat-sub">5-Fold Stratified Split</div>
          </div>
          <div class="stat-cell">
            <div class="stat-label">Holdout ROC</div>
            <div class="stat-num" id="statHoldoutRoc">0.8443</div>
            <div class="stat-sub">1,409 Unseen Records</div>
          </div>
          <div class="stat-cell">
            <div class="stat-label">Budget Limit</div>
            <div class="stat-num">15 / 15</div>
            <div class="stat-sub">Runs Completed</div>
          </div>
        </div>

      </section>

      <!-- ========================================== -->
      <!-- SUBSECTION 2: COMPLETE VISUALIZATIONS      -->
      <!-- ========================================== -->
      <section class="view-subsection" id="sub-trajectory">
        <div class="section-head">
          <div class="section-tag">01 // VISUALIZATIONS</div>
          <h2 class="section-title">Model Architecture & Trajectory Visualizations</h2>
          <div class="section-sub">Interactive multi-model tournament benchmarks, 15-run optimization curves, and deep-dive experiment telemetry.</div>
        </div>

        <!-- Chart Category Filter Bar -->
        <div class="chart-filter-bar">
          <button class="filter-btn active" id="filterBtn-all" onclick="filterChartSection('all')">All Visualizations (14)</button>
          <button class="filter-btn" id="filterBtn-tourney" onclick="filterChartSection('tourney')">Tournament Benchmark (4)</button>
          <button class="filter-btn" id="filterBtn-traj" onclick="filterChartSection('traj')">Optimization Trajectory (4)</button>
          <button class="filter-btn" id="filterBtn-inspector" onclick="filterChartSection('inspector')">Deep-Dive Inspector (6)</button>
        </div>

        <!-- GROUP 1: MULTI-MODEL TOURNAMENT -->
        <div id="chartGroup-tourney">
          <div class="chart-group-title">
            <span class="group-tag">TOURNAMENT</span>
            <span>Comparative Inductive Bias: 5 Architectural Families</span>
          </div>
          
          <!-- Tournament Cards 5-Grid -->
          <div class="tourney-cards-grid" id="tourneyCardsGrid"></div>

          <!-- Tournament 2x2 Charts Grid -->
          <div class="charts-grid-2">
            <div class="chart-card">
              <div class="chart-card-head">
                <span class="chart-heading">Predictive Capability: ROC-AUC vs PR-AUC Across Architectures</span>
                <span class="chart-badge">5 Families</span>
              </div>
              <div class="chart-canvas-wrapper">
                <canvas id="chartTourneyMetrics"></canvas>
              </div>
            </div>

            <div class="chart-card">
              <div class="chart-card-head">
                <span class="chart-heading">Overfitting Risk: Generalization Gap & Overfitting Index (&Omega;%)</span>
                <span class="chart-badge">Train vs Val Gap</span>
              </div>
              <div class="chart-canvas-wrapper">
                <canvas id="chartTourneyOverfit"></canvas>
              </div>
            </div>
          </div>

          <div class="charts-grid-2">
            <div class="chart-card">
              <div class="chart-card-head">
                <span class="chart-heading">Computational Efficiency: Training Time vs Inference Latency</span>
                <span class="chart-badge">Tesla T4 GPU</span>
              </div>
              <div class="chart-canvas-wrapper">
                <canvas id="chartTourneySpeed"></canvas>
              </div>
            </div>

            <div class="chart-card">
              <div class="chart-card-head">
                <span class="chart-heading">Throughput Comparison: Inference Samples / Sec</span>
                <span class="chart-badge">Holdout Batch</span>
              </div>
              <div class="chart-canvas-wrapper">
                <canvas id="chartTourneyThroughput"></canvas>
              </div>
            </div>
          </div>
        </div>

        <!-- GROUP 2: 15-RUN OPTIMIZATION TRAJECTORY -->
        <div id="chartGroup-traj">
          <div class="chart-group-title">
            <span class="group-tag">TRAJECTORY</span>
            <span>Autonomous Optimization & Hyperparameter Dynamics (Runs 1 &rarr; 15)</span>
          </div>

          <div class="charts-grid-2">
            <div class="chart-card">
              <div class="chart-card-head">
                <span class="chart-heading">Predictive Metric Capability Transformation</span>
                <span class="chart-badge">ROC-AUC & PR-AUC</span>
              </div>
              <div class="chart-canvas-wrapper">
                <canvas id="chartMetricProgression"></canvas>
              </div>
            </div>

            <div class="chart-card">
              <div class="chart-card-head">
                <span class="chart-heading">Generalization Gap & Overfitting Index (&Omega;%) Compression</span>
                <span class="chart-badge">Regularization</span>
              </div>
              <div class="chart-canvas-wrapper">
                <canvas id="chartOverfittingIndex"></canvas>
              </div>
            </div>
          </div>

          <div class="charts-grid-2">
            <div class="chart-card">
              <div class="chart-card-head">
                <span class="chart-heading">Loss Dynamics & Convergence Trajectory (Train vs Val Loss)</span>
                <span class="chart-badge">Cross-Entropy Log Loss</span>
              </div>
              <div class="chart-canvas-wrapper">
                <canvas id="chartLossTrajectories"></canvas>
              </div>
            </div>

            <div class="chart-card">
              <div class="chart-card-head">
                <span class="chart-heading">Hyperparameter Annealing: Depth & L2 Leaf Regularization</span>
                <span class="chart-badge">Dual-Axis Search</span>
              </div>
              <div class="chart-canvas-wrapper">
                <canvas id="chartHyperparamEvolution"></canvas>
              </div>
            </div>
          </div>

          <!-- Existing Duo: Iteration Trajectory & Holdout Benchmark -->
          <div class="charts-duo">
            <div class="chart-card">
              <div class="chart-card-head">
                <span class="chart-heading">Trajectory Across Iterations</span>
                <span class="chart-badge">ROC-AUC vs PR-AUC</span>
              </div>
              <div class="chart-canvas-wrapper">
                <canvas id="linearTrajChart"></canvas>
              </div>
            </div>

            <div class="chart-card">
              <div class="chart-card-head">
                <span class="chart-heading">Validation vs Holdout Benchmark</span>
                <span class="chart-badge">1,409 Test Rows</span>
              </div>
              <div class="chart-canvas-wrapper">
                <canvas id="linearHoldoutChart"></canvas>
              </div>
            </div>
          </div>

          <div class="insights-grid">
            <div class="insight-card">
              <div class="insight-title">Validation Stability</div>
              <div class="insight-body">Model ranking metrics remain tightly bounded between 0.8492 and 0.8506 across trials.</div>
            </div>
            <div class="insight-card">
              <div class="insight-title">Holdout Delta</div>
              <div class="insight-body">Cross-validation score of 0.8506 maps cleanly to 0.8443 on sealed holdout accounts.</div>
            </div>
            <div class="insight-card">
              <div class="insight-title">Regularization Gain</div>
              <div class="insight-body">L2 Regularization up to 14.0 reduced overfitting index &Omega; from 11.27% down to 6.88%.</div>
            </div>
          </div>
        </div>

        <!-- GROUP 3: DEEP-DIVE EXPERIMENT INSPECTOR -->
        <div id="chartGroup-inspector">
          <div class="chart-group-title">
            <span class="group-tag">INSPECTOR</span>
            <span>Deep-Dive Diagnostic Telemetry (Iterative Stepper)</span>
          </div>

          <!-- 15-Run Stepper Bar -->
          <div class="stepper-bar" id="stepperBar"></div>

          <!-- Experiment Hypothesis & Formulation Card -->
          <div class="exp-meta-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
              <span id="expBadge" class="chart-badge" style="color: var(--accent-cyan); font-weight: 600;">Experiment #12 Selected</span>
              <span id="expFamTag" class="chart-badge">CATBOOST</span>
            </div>
            <div id="expHypothesis" style="font-size: 13.5px; font-weight: 600; color: var(--text-high); margin-bottom: 6px;"></div>
            <div id="expMath" style="font-size: 12.5px; color: var(--text-medium); line-height: 1.55;"></div>
            <div id="expConfigTags" style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px;"></div>
          </div>

          <!-- 6-Chart Responsive Grid (3x2) -->
          <div class="charts-grid-3">
            <div class="chart-card">
              <div class="chart-card-head">
                <span class="chart-heading">Loss Convergence Curve</span>
                <span class="chart-badge">Iterations</span>
              </div>
              <div class="chart-canvas-wrapper-sm">
                <canvas id="chartExpLoss"></canvas>
              </div>
            </div>

            <div class="chart-card">
              <div class="chart-card-head">
                <span class="chart-heading">ROC Curve (FPR vs TPR)</span>
                <span class="chart-badge">Discrimination</span>
              </div>
              <div class="chart-canvas-wrapper-sm">
                <canvas id="chartExpROC"></canvas>
              </div>
            </div>

            <div class="chart-card">
              <div class="chart-card-head">
                <span class="chart-heading">Precision-Recall Curve</span>
                <span class="chart-badge">Imbalance</span>
              </div>
              <div class="chart-canvas-wrapper-sm">
                <canvas id="chartExpPR"></canvas>
              </div>
            </div>
          </div>

          <div class="charts-grid-3">
            <div class="chart-card">
              <div class="chart-card-head">
                <span class="chart-heading">Probability Calibration</span>
                <span class="chart-badge">Reliability</span>
              </div>
              <div class="chart-canvas-wrapper-sm">
                <canvas id="chartExpCalib"></canvas>
              </div>
            </div>

            <div class="chart-card">
              <div class="chart-card-head">
                <span class="chart-heading">Prediction Confidence Density</span>
                <span class="chart-badge">Histogram</span>
              </div>
              <div class="chart-canvas-wrapper-sm">
                <canvas id="chartExpHist"></canvas>
              </div>
            </div>

            <div class="chart-card">
              <div class="chart-card-head">
                <span class="chart-heading">Top Feature Importances</span>
                <span class="chart-badge">TreeSHAP</span>
              </div>
              <div class="chart-canvas-wrapper-sm">
                <canvas id="chartExpFeats"></canvas>
              </div>
            </div>
          </div>

          <!-- Fold Performance & Confusion Matrix Breakdown -->
          <div class="exp-meta-card">
            <div style="font-size: 13px; font-weight: 600; color: var(--text-high); margin-bottom: 8px;">5-Fold Cross-Validation Metrics Breakdown</div>
            <div id="expFoldTable" style="font-family: var(--font-mono); font-size: 12px; line-height: 1.6; color: var(--text-medium);"></div>
            <div class="cm-grid">
              <div class="cm-card" id="expCM05"></div>
              <div class="cm-card" id="expCMOpt"></div>
            </div>
          </div>

        </div>

      </section>

      <!-- ========================================== -->
      <!-- SUBSECTION 3: PARAMETER MATRIX             -->
      <!-- ========================================== -->
      <section class="view-subsection" id="sub-params">
        <div class="section-head">
          <div class="section-tag">02 // ARCHITECTURE</div>
          <h2 class="section-title">Classifier & Hyperparameter Matrix</h2>
          <div class="section-sub">Direct mathematical parameter comparison against the top-performing Champion model.</div>
        </div>

        <div style="margin-bottom:24px; display:flex; align-items:center; gap:16px;">
          <span style="font-size:13px; color:var(--text-medium);">Select Trial to Compare:</span>
          <select id="paramSelect" onchange="renderParamComparison()" style="padding:10px 18px; border:1px solid var(--border-medium); border-radius:8px; background:var(--bg-surface); font-family:var(--font-sans); font-size:12px; color:var(--text-high); outline:none;">
          </select>
        </div>

        <div class="params-comparison-grid" id="paramsComparisonWrap">
        </div>
      </section>

      <!-- ========================================== -->
      <!-- SUBSECTION 4: HOLDOUT VERIFICATION         -->
      <!-- ========================================== -->
      <section class="view-subsection" id="sub-holdout">
        <div class="section-head">
          <div class="section-tag">03 // SEALED VERIFICATION</div>
          <h2 class="section-title">Holdout Test & Decision Policy</h2>
          <div class="section-sub">Evaluated on 1,409 unseen accounts with cost-optimized decision thresholds.</div>
        </div>

        <div class="holdout-panel">
          <div class="stat-strip" style="margin-bottom: 40px;">
            <div class="stat-cell">
              <div class="stat-label">Holdout Samples</div>
              <div class="stat-num">1,409</div>
              <div class="stat-sub">20% Sealed Split</div>
            </div>
            <div class="stat-cell">
              <div class="stat-label">Holdout ROC-AUC</div>
              <div class="stat-num">0.8443</div>
              <div class="stat-sub">Ranking Quality</div>
            </div>
            <div class="stat-cell">
              <div class="stat-label">Holdout PR-AUC</div>
              <div class="stat-num">0.6628</div>
              <div class="stat-sub">2.5x Over Baseline</div>
            </div>
            <div class="stat-cell">
              <div class="stat-label">Holdout Log Loss</div>
              <div class="stat-num">0.4736</div>
              <div class="stat-sub">Calibrated Probability</div>
            </div>
          </div>

          <div style="font-size:14px; font-weight:600; color:var(--text-high); margin-bottom:8px;">Operational Cutoff Comparison</div>
          <div style="font-size:12.5px; color:var(--text-medium); line-height:1.55; margin-bottom:16px;">
            The optimal 0.490 cutoff (Youden J* index) captures 82.5% of departing subscribers with 51.6% precision, minimizing unnecessary retention credits while controlling churn exposure.
          </div>

          <table class="policy-table">
            <thead>
              <tr>
                <th>Strategy</th>
                <th>Cutoff (&tau;)</th>
                <th>Recall</th>
                <th>Precision</th>
                <th>Operational Impact</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Conservative</td>
                <td style="font-family:var(--font-mono);">0.700</td>
                <td style="font-family:var(--font-mono);">58.3%</td>
                <td style="font-family:var(--font-mono);">65.2%</td>
                <td>Targets only high-confidence departing accounts; minimizes credit expenditure.</td>
              </tr>
              <tr class="highlight-policy">
                <td>Optimal (Recommended)</td>
                <td style="font-family:var(--font-mono); color:var(--accent-emerald);">0.490</td>
                <td style="font-family:var(--font-mono); color:var(--accent-emerald);">82.5%</td>
                <td style="font-family:var(--font-mono); color:var(--accent-emerald);">51.6%</td>
                <td>Youden J* optimal balance: catches 82.5% of departures with lowest overall cost.</td>
              </tr>
              <tr>
                <td>Aggressive</td>
                <td style="font-family:var(--font-mono);">0.300</td>
                <td style="font-family:var(--font-mono);">91.7%</td>
                <td style="font-family:var(--font-mono);">43.1%</td>
                <td>High recall tier; captures over 90% of churners but increases intervention cost.</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <!-- ========================================== -->
      <!-- SUBSECTION 5: COMPLETE LEDGER              -->
      <!-- ========================================== -->
      <section class="view-subsection" id="sub-ledger">
        <div class="section-head">
          <div class="section-tag">04 // AUDIT TRAIL</div>
          <h2 class="section-title">Experiment Ledger</h2>
          <div class="section-sub">Chronological register of all 15 evaluated configurations and cross-validation checkpoints.</div>
        </div>

        <div class="ledger-table-wrap">
          <table class="ledger-table">
            <thead>
              <tr>
                <th>Run</th>
                <th>Family</th>
                <th>Depth</th>
                <th>LR</th>
                <th>L2</th>
                <th>Iters</th>
                <th>Val ROC</th>
                <th>Val PR</th>
                <th>Gap</th>
                <th>Hypothesis / Takeaway</th>
              </tr>
            </thead>
            <tbody id="ledgerTableBody">
            </tbody>
          </table>
        </div>
      </section>


    </main>
  </div>

  <footer>
    <div class="container">
      Linear • Autonomous Machine Learning Intelligence System • IBM Telco Customer Churn
    </div>
  </footer>

  <script>
    // ========================================================
    // 3D FLOATING CUBES & WIREFRAME INTERACTIVE BACKGROUND
    // Directly inspired by dark geometric 3D imagery
    // ========================================================
    (function init3DBackground() {{
      const canvas = document.getElementById("webgl-canvas");
      const glow = document.getElementById("cursorGlow");
      if (!canvas) return;

      let mouseX = 0;
      let mouseY = 0;
      let targetMouseX = 0;
      let targetMouseY = 0;
      let clickPulse = 0;

      // Track cursor position
      window.addEventListener("mousemove", (e) => {{
        const x = e.clientX;
        const y = e.clientY;
        targetMouseX = (x / window.innerWidth) * 2 - 1;
        targetMouseY = -(y / window.innerHeight) * 2 + 1;

        if (glow) {{
          glow.style.left = `${{x}}px`;
          glow.style.top = `${{y}}px`;
        }}
      }});

      // Interactive Click Shockwave
      window.addEventListener("mousedown", () => {{
        clickPulse = 1.0;
        if (glow) {{
          glow.style.width = "750px";
          glow.style.height = "750px";
        }}
      }});

      window.addEventListener("mouseup", () => {{
        if (glow) {{
          glow.style.width = "580px";
          glow.style.height = "580px";
        }}
      }});

      // Check for Three.js
      if (typeof THREE !== "undefined") {{
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.z = 160;

        const renderer = new THREE.WebGLRenderer({{
          canvas: canvas,
          alpha: true,
          antialias: true,
          powerPreference: "high-performance"
        }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

        // Ambient & Directional Lights
        const ambientLight = new THREE.AmbientLight(0x181c2b, 1.4);
        scene.add(ambientLight);

        const dirLight = new THREE.DirectionalLight(0x7c8ba1, 1.2);
        dirLight.position.set(100, 150, 100);
        scene.add(dirLight);

        // Interactive Cursor Point Light (soft indigo / cyan)
        const mouseLight = new THREE.PointLight(0x5e6ad2, 2.8, 320);
        mouseLight.position.set(0, 0, 80);
        scene.add(mouseLight);

        // Create ~32 Floating 3D Cubes (Solid Dark Slate + Wireframe Edges)
        const cubes = [];
        const cubeCount = 28;

        const solidMat = new THREE.MeshStandardMaterial({{
          color: 0x111318,
          roughness: 0.35,
          metalness: 0.85
        }});

        const wireframeColors = [0x5e6ad2, 0x4ebe9e, 0x8a8f98, 0x38bdf8, 0x2e3440];

        for (let i = 0; i < cubeCount; i++) {{
          const size = Math.random() * 18 + 10;
          const geom = new THREE.BoxGeometry(size, size, size);
          const edges = new THREE.EdgesGeometry(geom);

          const group = new THREE.Group();

          if (i % 3 !== 0) {{
            const mesh = new THREE.Mesh(geom, solidMat);
            group.add(mesh);

            const lineMat = new THREE.LineBasicMaterial({{
              color: 0x2e3440,
              transparent: true,
              opacity: 0.6
            }});
            const wire = new THREE.LineSegments(edges, lineMat);
            group.add(wire);
          }} else {{
            const color = wireframeColors[i % wireframeColors.length];
            const lineMat = new THREE.LineBasicMaterial({{
              color: color,
              transparent: true,
              opacity: 0.55
            }});
            const wire = new THREE.LineSegments(edges, lineMat);
            group.add(wire);
          }}

          const x = (Math.random() - 0.5) * 260;
          const y = (Math.random() - 0.5) * 200;
          const z = (Math.random() - 0.5) * 160 - 20;

          group.position.set(x, y, z);
          scene.add(group);

          cubes.push({{
            group: group,
            baseX: x,
            baseY: y,
            baseZ: z,
            rotSpeedX: (Math.random() - 0.5) * 0.008,
            rotSpeedY: (Math.random() - 0.5) * 0.010,
            rotSpeedZ: (Math.random() - 0.5) * 0.006,
            phase: Math.random() * Math.PI * 2,
            floatSpeed: Math.random() * 0.008 + 0.004,
            floatAmp: Math.random() * 12 + 6
          }});
        }}

        let time = 0;
        function animate() {{
          requestAnimationFrame(animate);
          time += 0.016;

          mouseX += (targetMouseX - mouseX) * 0.04;
          mouseY += (targetMouseY - mouseY) * 0.04;

          camera.position.x = mouseX * 22;
          camera.position.y = mouseY * 16;
          camera.lookAt(0, 0, 0);

          mouseLight.position.x = mouseX * 120;
          mouseLight.position.y = mouseY * 90;

          if (clickPulse > 0) {{
            clickPulse *= 0.94;
            if (clickPulse < 0.01) clickPulse = 0;
          }}

          for (let i = 0; i < cubes.length; i++) {{
            const c = cubes[i];
            const g = c.group;

            const boost = 1 + clickPulse * 4;
            g.rotation.x += c.rotSpeedX * boost;
            g.rotation.y += c.rotSpeedY * boost;
            g.rotation.z += c.rotSpeedZ * boost;

            g.position.y = c.baseY + Math.sin(time * c.floatSpeed * 60 + c.phase) * c.floatAmp;

            const dx = g.position.x - mouseLight.position.x;
            const dy = g.position.y - mouseLight.position.y;
            const dist = Math.sqrt(dx * dx + dy * dy);

            if (dist < 70) {{
              const force = (1 - dist / 70) * 14;
              g.position.x += (dx / dist) * force * 0.15;
              g.position.y += (dy / dist) * force * 0.15;
            }} else {{
              g.position.x += (c.baseX - g.position.x) * 0.03;
            }}
          }}

          renderer.render(scene, camera);
        }}

        animate();

        window.addEventListener("resize", () => {{
          camera.aspect = window.innerWidth / window.innerHeight;
          camera.updateProjectionMatrix();
          renderer.setSize(window.innerWidth, window.innerHeight);
        }});

      }} else {{
        const ctx = canvas.getContext("2d");
        function resizeFallback() {{
          canvas.width = window.innerWidth;
          canvas.height = window.innerHeight;
        }}
        resizeFallback();
        window.addEventListener("resize", resizeFallback);
      }}
    }})();

    // ========================================================
    // AUTHENTIC TRAINING ARTIFACTS (DIRECT FROM RUN OUTPUTS)
    // ========================================================
    const runs = {runs_json};
    const holdout = {holdout_json};
    const supervisorReport = {supervisor_json_str};

    const modelMap = {{
      'catboost': 'CatBoost',
      'lightgbm': 'LightGBM',
      'xgboost': 'XGBoost',
      'random_forest': 'RandomForest',
      'logistic_regression': 'LogisticRegression'
    }};

    // Dynamically derive experimentData from genuine ledger runs
    const experimentData = runs.map(r => {{
      const hp = r.hyperparameters || {{}};
      const met = r.metrics || {{}};
      const famRaw = (hp.model_family || 'catboost').toLowerCase();
      const modelName = modelMap[famRaw] || hp.model_family || 'CatBoost';
      return {{
        run_id: r.experiment_id,
        model: modelName,
        depth: hp.depth || 6,
        lr: hp.learning_rate || 0.05,
        l2: hp.l2_leaf_reg || 3.0,
        iters: hp.iterations || 1000,
        bag: hp.bagging_temperature !== undefined ? hp.bagging_temperature : 0.2,
        roc: Number((met.oof_roc_auc || 0).toFixed(4)),
        pr: Number((met.oof_pr_auc || 0).toFixed(4)),
        loss: Number((met.oof_log_loss || 0).toFixed(4)),
        gap: Number((met.generalization_gap || 0).toFixed(4)),
        takeaway: r.hypothesis || `Run #${{r.experiment_id}} (${{modelName}})`
      }};
    }});

    function getChampion() {{
      const targetId = holdout.champion_experiment_id || 12;
      return experimentData.find(e => e.run_id === targetId) ||
        experimentData.reduce((prev, curr) => (curr.roc > prev.roc ? curr : prev), experimentData[0]);
    }}

    let currentStepIndex = 1;
    let selectedRunIdx = 11; // Index 11 corresponds to Experiment #12 (Champion)
    let expCharts = {{}};
    let tournamentChartsInitialized = false;
    let overviewChartsInitialized = false;

    window.addEventListener("DOMContentLoaded", () => {{
      const champ = getChampion();
      document.getElementById("statBestModel").textContent = champ.model;
      document.getElementById("statBestRun").textContent = `Run #${{champ.run_id}} (Champion)`;
      document.getElementById("statTopVal").textContent = (holdout.oof_roc_auc ? holdout.oof_roc_auc.toFixed(4) : champ.roc.toFixed(4));
      document.getElementById("statHoldoutRoc").textContent = (holdout.holdout_roc_auc ? holdout.holdout_roc_auc.toFixed(4) : "0.8443");

      renderLedgerTable();
      populateParamSelect();
      renderParamComparison();
      initLinearCharts();

      // Initialize all visualization graphs
      populateTournamentTab();
      renderOverviewCharts();
      buildStepper();
      renderSelectedRun(selectedRunIdx);
    }});

    function switchView(tabKey) {{
      document.querySelectorAll(".sub-tab").forEach(tab => {{
        tab.classList.toggle("active", tab.id === `tab-${{tabKey}}`);
      }});
      document.querySelectorAll(".view-subsection").forEach(sec => {{
        sec.classList.toggle("active", sec.id === `sub-${{tabKey}}`);
      }});
      window.scrollTo({{ top: 0, behavior: 'smooth' }});

      if (tabKey === 'trajectory') {{
        setTimeout(() => {{
          window.dispatchEvent(new Event('resize'));
        }}, 60);
      }}
    }}

    function filterChartSection(key) {{
      document.querySelectorAll('.filter-btn').forEach(btn => {{
        btn.classList.toggle('active', btn.id === `filterBtn-${{key}}`);
      }});

      const tourneySec = document.getElementById('chartGroup-tourney');
      const trajSec = document.getElementById('chartGroup-traj');
      const inspSec = document.getElementById('chartGroup-inspector');

      if (key === 'all') {{
        tourneySec.style.display = 'block';
        trajSec.style.display = 'block';
        inspSec.style.display = 'block';
      }} else if (key === 'tourney') {{
        tourneySec.style.display = 'block';
        trajSec.style.display = 'none';
        inspSec.style.display = 'none';
      }} else if (key === 'traj') {{
        tourneySec.style.display = 'none';
        trajSec.style.display = 'block';
        inspSec.style.display = 'none';
      }} else if (key === 'inspector') {{
        tourneySec.style.display = 'none';
        trajSec.style.display = 'none';
        inspSec.style.display = 'block';
      }}

      window.dispatchEvent(new Event('resize'));
    }}

    function stepRun() {{
      currentStepIndex = (currentStepIndex % experimentData.length) + 1;
      document.getElementById("stepActionText").textContent = `Step Run (${{currentStepIndex}}/${{experimentData.length}})`;
      
      const select = document.getElementById("paramSelect");
      if (select) {{
        const champ = getChampion();
        select.value = currentStepIndex === champ.run_id ? (champ.run_id === 1 ? 2 : 1) : currentStepIndex;
        renderParamComparison();
      }}

      // Also sync deep-dive inspector to currentStepIndex
      renderSelectedRun(currentStepIndex - 1);
      switchView('params');
    }}

    function populateParamSelect() {{
      const select = document.getElementById("paramSelect");
      if (!select) return;
      select.innerHTML = "";
      const champ = getChampion();
      experimentData.forEach(e => {{
        if (e.run_id !== champ.run_id) {{
          const opt = document.createElement("option");
          opt.value = e.run_id;
          opt.textContent = `Run #${{e.run_id}} (${{e.model}}) — Depth ${{e.depth}}, LR ${{e.lr}}, L2 ${{e.l2}}`;
          select.appendChild(opt);
        }}
      }});
    }}

    function renderParamComparison() {{
      const champ = getChampion();
      const select = document.getElementById("paramSelect");
      const targetId = parseInt(select ? select.value : (champ.run_id === 1 ? 2 : 1), 10);
      const target = experimentData.find(e => e.run_id === targetId) || experimentData[1];

      const wrap = document.getElementById("paramsComparisonWrap");
      if (!wrap) return;

      wrap.innerHTML = `
        <div class="param-box champion">
          <div class="param-box-head">
            <div>
              <div style="font-size:14px; font-weight:600; color:var(--text-high);">Champion: ${{champ.model}}</div>
              <div style="font-size:11.5px; color:var(--text-medium); margin-top:2px;">Run #${{champ.run_id}} • Top Evaluated</div>
            </div>
            <span class="param-pill champ">CHAMPION</span>
          </div>
          <div class="param-spec-row"><span class="spec-key">Model Family</span><span class="spec-val">${{champ.model}}</span></div>
          <div class="param-spec-row"><span class="spec-key">Tree Depth / Leaves</span><span class="spec-val">${{champ.depth}}</span></div>
          <div class="param-spec-row"><span class="spec-key">Learning Rate</span><span class="spec-val">${{champ.lr}}</span></div>
          <div class="param-spec-row"><span class="spec-key">L2 Regularization</span><span class="spec-val">${{champ.l2}}</span></div>
          <div class="param-spec-row"><span class="spec-key">Iterations</span><span class="spec-val">${{champ.iters}}</span></div>
          <div class="param-spec-row"><span class="spec-key">Bagging Temp</span><span class="spec-val">${{champ.bag}}</span></div>
          <div class="param-spec-row"><span class="spec-key">Validation ROC-AUC</span><span class="spec-val" style="color:var(--accent-emerald); font-weight:600;">${{champ.roc.toFixed(4)}}</span></div>
          <div class="param-spec-row"><span class="spec-key">Validation PR-AUC</span><span class="spec-val">${{champ.pr.toFixed(4)}}</span></div>
          <div class="param-spec-row"><span class="spec-key">Generalization Gap</span><span class="spec-val">${{champ.gap.toFixed(4)}}</span></div>
        </div>

        <div class="param-box">
          <div class="param-box-head">
            <div>
              <div style="font-size:14px; font-weight:600; color:var(--text-high);">Contender: ${{target.model}}</div>
              <div style="font-size:11.5px; color:var(--text-medium); margin-top:2px;">Run #${{target.run_id}} • Comparison Trial</div>
            </div>
            <span class="param-pill trial">TRIAL</span>
          </div>
          <div class="param-spec-row"><span class="spec-key">Model Family</span><span class="spec-val">${{target.model}}</span></div>
          <div class="param-spec-row"><span class="spec-key">Tree Depth / Leaves</span><span class="spec-val">${{target.depth}}</span></div>
          <div class="param-spec-row"><span class="spec-key">Learning Rate</span><span class="spec-val">${{target.lr}}</span></div>
          <div class="param-spec-row"><span class="spec-key">L2 Regularization</span><span class="spec-val">${{target.l2}}</span></div>
          <div class="param-spec-row"><span class="spec-key">Iterations</span><span class="spec-val">${{target.iters}}</span></div>
          <div class="param-spec-row"><span class="spec-key">Bagging Temp</span><span class="spec-val">${{target.bag}}</span></div>
          <div class="param-spec-row"><span class="spec-key">Validation ROC-AUC</span><span class="spec-val">${{target.roc.toFixed(4)}}</span></div>
          <div class="param-spec-row"><span class="spec-key">Validation PR-AUC</span><span class="spec-val">${{target.pr.toFixed(4)}}</span></div>
          <div class="param-spec-row"><span class="spec-key">Generalization Gap</span><span class="spec-val">${{target.gap.toFixed(4)}}</span></div>
        </div>
      `;
    }}

    function renderLedgerTable() {{
      const tbody = document.getElementById("ledgerTableBody");
      if (!tbody) return;
      tbody.innerHTML = "";
      const champ = getChampion();

      experimentData.forEach(e => {{
        const isChamp = (e.run_id === champ.run_id);
        const tr = document.createElement("tr");
        if (isChamp) tr.className = "champ-row";
        tr.onclick = () => {{
          const select = document.getElementById("paramSelect");
          if (select) {{
            select.value = isChamp ? (champ.run_id === 1 ? 2 : 1) : e.run_id;
            renderParamComparison();
          }}
          renderSelectedRun(e.run_id - 1);
          switchView('params');
        }};

        tr.innerHTML = `
          <td><strong>#${{e.run_id}}</strong> ${{isChamp ? '<span style="font-size:9.5px; background:var(--accent-emerald-dim); color:var(--accent-emerald); border:1px solid rgba(78, 190, 158, 0.3); padding:2px 5px; border-radius:4px; margin-left:6px;">BEST</span>' : ''}}</td>
          <td><span style="color:var(--text-high); font-family:var(--font-sans);">${{e.model}}</span></td>
          <td>${{e.depth}}</td>
          <td>${{e.lr}}</td>
          <td>${{e.l2}}</td>
          <td>${{e.iters}}</td>
          <td><strong style="color:var(--text-high);">${{e.roc.toFixed(4)}}</strong></td>
          <td>${{e.pr.toFixed(4)}}</td>
          <td>${{e.gap.toFixed(4)}}</td>
          <td style="font-family:var(--font-sans); font-size:11px; color:var(--text-medium); max-width:240px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">${{e.takeaway}}</td>
        `;
        tbody.appendChild(tr);
      }});
    }}

    // ========================================================
    // CHART INITIALIZATIONS
    // ========================================================

    // 1. Tournament Cards & 4 Tournament Charts
    function populateTournamentTab() {{
      const tourneyRuns = (supervisorReport.tournament_leaderboard && supervisorReport.tournament_leaderboard.length > 0)
        ? supervisorReport.tournament_leaderboard
        : runs.slice(0, 5);

      const sorted = [...tourneyRuns].sort((a, b) => b.metrics.oof_roc_auc - a.metrics.oof_roc_auc);
      const grid = document.getElementById('tourneyCardsGrid');
      if (grid) {{
        grid.innerHTML = '';
        sorted.forEach((item, idx) => {{
          const famRaw = (item.hyperparameters.model_family || 'catboost').toLowerCase();
          const fam = modelMap[famRaw] || item.hyperparameters.model_family;
          const met = item.metrics;
          const isWinner = (idx === 0);

          const card = document.createElement('div');
          card.className = 'tourney-card' + (isWinner ? ' champ' : '');
          card.innerHTML = `
            <div class="tourney-rank">${{isWinner ? '&#127942; TOURNAMENT CHAMPION' : 'RANK #' + (idx + 1)}}</div>
            <div class="tourney-name">${{fam}}</div>
            <div class="tourney-stat-row"><span>OOF ROC-AUC</span><span class="tourney-stat-val" style="color:var(--accent-emerald);">${{met.oof_roc_auc.toFixed(5)}}</span></div>
            <div class="tourney-stat-row"><span>OOF PR-AUC</span><span class="tourney-stat-val">${{met.oof_pr_auc.toFixed(4)}}</span></div>
            <div class="tourney-stat-row"><span>Gen. Gap</span><span class="tourney-stat-val">${{met.generalization_gap.toFixed(4)}}</span></div>
            <div class="tourney-stat-row"><span>Training Time</span><span class="tourney-stat-val">${{(met.training_duration_seconds || 0).toFixed(1)}}s</span></div>
            <div class="tourney-stat-row"><span>Throughput</span><span class="tourney-stat-val">${{Math.round(met.throughput_samples_per_sec || 0)}} /s</span></div>
          `;
          grid.appendChild(card);
        }});
      }}

      const famLabels = sorted.map(s => {{
        const r = (s.hyperparameters.model_family || '').toLowerCase();
        return modelMap[r] || s.hyperparameters.model_family.toUpperCase();
      }});
      const rocs = sorted.map(s => s.metrics.oof_roc_auc);
      const prs = sorted.map(s => s.metrics.oof_pr_auc);
      const gaps = sorted.map(s => s.metrics.generalization_gap);
      const omegas = sorted.map(s => s.metrics.overfitting_index);
      const fitTimes = sorted.map(s => s.metrics.training_duration_seconds);
      const latencies = sorted.map(s => s.metrics.inference_latency_ms);
      const throughputs = sorted.map(s => s.metrics.throughput_samples_per_sec || 0);

      // Chart 1: Predictive Capability
      const ctx1 = document.getElementById('chartTourneyMetrics');
      if (ctx1) {{
        new Chart(ctx1.getContext('2d'), {{
          type: 'bar',
          data: {{
            labels: famLabels,
            datasets: [
              {{ label: 'OOF ROC-AUC', data: rocs, backgroundColor: '#38BDF8', borderRadius: 4 }},
              {{ label: 'OOF PR-AUC', data: prs, backgroundColor: '#4EBE9E', borderRadius: 4 }}
            ]
          }},
          options: {{
            responsive: true,
            maintainAspectRatio: false,
            scales: {{
              x: {{ ticks: {{ color: '#8A8F98' }}, grid: {{ display: false }} }},
              y: {{ min: 0.60, max: 0.88, ticks: {{ color: '#8A8F98' }}, grid: {{ color: 'rgba(255, 255, 255, 0.04)' }} }}
            }}
          }}
        }});
      }}

      // Chart 2: Overfitting Risk
      const ctx2 = document.getElementById('chartTourneyOverfit');
      if (ctx2) {{
        new Chart(ctx2.getContext('2d'), {{
          type: 'bar',
          data: {{
            labels: famLabels,
            datasets: [
              {{ label: 'Overfitting Index (Ω %)', data: omegas, backgroundColor: omegas.map(v => v > 25 ? '#F43F5E' : (v < 12 ? '#4EBE9E' : '#F59E0B')), borderRadius: 4 }},
              {{ label: 'Generalization Gap (x100)', data: gaps.map(g => g * 100), backgroundColor: '#5E6AD2', borderRadius: 4 }}
            ]
          }},
          options: {{
            responsive: true,
            maintainAspectRatio: false,
            scales: {{
              x: {{ ticks: {{ color: '#8A8F98' }}, grid: {{ display: false }} }},
              y: {{ ticks: {{ color: '#8A8F98' }}, grid: {{ color: 'rgba(255, 255, 255, 0.04)' }} }}
            }}
          }}
        }});
      }}

      // Chart 3: Speed & Latency
      const ctx3 = document.getElementById('chartTourneySpeed');
      if (ctx3) {{
        new Chart(ctx3.getContext('2d'), {{
          type: 'bar',
          data: {{
            labels: famLabels,
            datasets: [
              {{ label: 'Training Time (sec)', data: fitTimes, backgroundColor: '#F43F5E', yAxisID: 'yTime', borderRadius: 4 }},
              {{ label: 'Inference Latency (ms)', data: latencies, backgroundColor: '#38BDF8', yAxisID: 'yLat', borderRadius: 4 }}
            ]
          }},
          options: {{
            responsive: true,
            maintainAspectRatio: false,
            scales: {{
              x: {{ ticks: {{ color: '#8A8F98' }}, grid: {{ display: false }} }},
              yTime: {{ type: 'linear', position: 'left', ticks: {{ color: '#F43F5E' }}, grid: {{ color: 'rgba(255, 255, 255, 0.04)' }} }},
              yLat: {{ type: 'linear', position: 'right', ticks: {{ color: '#38BDF8' }}, grid: {{ display: false }} }}
            }}
          }}
        }});
      }}

      // Chart 4: Throughput
      const ctx4 = document.getElementById('chartTourneyThroughput');
      if (ctx4) {{
        new Chart(ctx4.getContext('2d'), {{
          type: 'bar',
          data: {{
            labels: famLabels,
            datasets: [{{ label: 'Throughput (Samples / Sec)', data: throughputs, backgroundColor: '#F59E0B', borderRadius: 4 }}]
          }},
          options: {{
            responsive: true,
            maintainAspectRatio: false,
            scales: {{
              x: {{ ticks: {{ color: '#8A8F98' }}, grid: {{ display: false }} }},
              y: {{ ticks: {{ color: '#8A8F98' }}, grid: {{ color: 'rgba(255, 255, 255, 0.04)' }} }}
            }}
          }}
        }});
      }}
    }}

    // 2. Trajectory 4-Charts Suite
    function renderOverviewCharts() {{
      const labels = runs.map(r => 'Exp ' + r.experiment_id + ' (' + (r.hyperparameters.model_family || 'cb').substring(0, 2).toUpperCase() + ')');
      const rocData = runs.map(r => r.metrics.oof_roc_auc);
      const prData = runs.map(r => r.metrics.oof_pr_auc);
      const omegaData = runs.map(r => r.metrics.overfitting_index);
      const trainLossData = runs.map(r => r.metrics.final_train_loss);
      const valLossData = runs.map(r => r.metrics.final_val_loss);

      // Chart 5: Metric Progression
      const ctx5 = document.getElementById('chartMetricProgression');
      if (ctx5) {{
        new Chart(ctx5.getContext('2d'), {{
          type: 'line',
          data: {{
            labels: labels,
            datasets: [
              {{
                label: 'OOF ROC-AUC (Primary)',
                data: rocData,
                borderColor: '#38BDF8',
                backgroundColor: 'rgba(56, 189, 248, 0.08)',
                tension: 0.25,
                fill: true,
                borderWidth: 2
              }},
              {{
                label: 'OOF PR-AUC (Secondary)',
                data: prData,
                borderColor: '#4EBE9E',
                backgroundColor: 'transparent',
                tension: 0.25,
                borderWidth: 2,
                borderDash: [4, 4]
              }},
              {{
                label: 'Sealed Holdout Benchmark (0.8443)',
                data: Array(runs.length).fill(holdout.holdout_roc_auc || 0.84426),
                borderColor: '#5E6AD2',
                borderWidth: 1.5,
                borderDash: [2, 2],
                pointRadius: 0
              }}
            ]
          }},
          options: {{
            responsive: true,
            maintainAspectRatio: false,
            scales: {{
              x: {{ ticks: {{ color: '#8A8F98' }}, grid: {{ display: false }} }},
              y: {{ ticks: {{ color: '#8A8F98' }}, grid: {{ color: 'rgba(255, 255, 255, 0.04)' }}, min: 0.65, max: 0.86 }}
            }}
          }}
        }});
      }}

      // Chart 6: Overfitting Index
      const ctx6 = document.getElementById('chartOverfittingIndex');
      if (ctx6) {{
        new Chart(ctx6.getContext('2d'), {{
          type: 'bar',
          data: {{
            labels: labels,
            datasets: [{{
              label: 'Overfitting Index (Ω %)',
              data: omegaData,
              backgroundColor: omegaData.map(val => val > 20 ? '#F43F5E' : (val < 8 ? '#4EBE9E' : '#38BDF8')),
              borderRadius: 4
            }}]
          }},
          options: {{
            responsive: true,
            maintainAspectRatio: false,
            scales: {{
              x: {{ ticks: {{ color: '#8A8F98' }}, grid: {{ display: false }} }},
              y: {{ ticks: {{ color: '#8A8F98' }}, grid: {{ color: 'rgba(255, 255, 255, 0.04)' }} }}
            }}
          }}
        }});
      }}

      // Chart 7: Loss Dynamics
      const ctx7 = document.getElementById('chartLossTrajectories');
      if (ctx7) {{
        new Chart(ctx7.getContext('2d'), {{
          type: 'line',
          data: {{
            labels: labels,
            datasets: [
              {{
                label: 'Final Validation Loss',
                data: valLossData,
                borderColor: '#F59E0B',
                backgroundColor: 'transparent',
                tension: 0.25,
                borderWidth: 2
              }},
              {{
                label: 'Final Training Loss',
                data: trainLossData,
                borderColor: '#38BDF8',
                backgroundColor: 'transparent',
                tension: 0.25,
                borderWidth: 2
              }}
            ]
          }},
          options: {{
            responsive: true,
            maintainAspectRatio: false,
            scales: {{
              x: {{ ticks: {{ color: '#8A8F98' }}, grid: {{ display: false }} }},
              y: {{ ticks: {{ color: '#8A8F98' }}, grid: {{ color: 'rgba(255, 255, 255, 0.04)' }} }}
            }}
          }}
        }});
      }}

      // Chart 8: Hyperparameter Annealing
      const depths = runs.map(r => r.hyperparameters.depth || 6);
      const l2s = runs.map(r => r.hyperparameters.l2_leaf_reg || 3.0);
      const ctx8 = document.getElementById('chartHyperparamEvolution');
      if (ctx8) {{
        new Chart(ctx8.getContext('2d'), {{
          type: 'line',
          data: {{
            labels: labels,
            datasets: [
              {{
                label: 'Tree Depth',
                data: depths,
                borderColor: '#5E6AD2',
                yAxisID: 'yDepth',
                borderWidth: 2
              }},
              {{
                label: 'L2 Leaf Regularization',
                data: l2s,
                borderColor: '#F43F5E',
                yAxisID: 'yL2',
                borderWidth: 2
              }}
            ]
          }},
          options: {{
            responsive: true,
            maintainAspectRatio: false,
            scales: {{
              x: {{ ticks: {{ color: '#8A8F98' }}, grid: {{ display: false }} }},
              yDepth: {{ type: 'linear', position: 'left', min: 3, max: 9, ticks: {{ color: '#5E6AD2' }}, grid: {{ color: 'rgba(255, 255, 255, 0.04)' }} }},
              yL2: {{ type: 'linear', position: 'right', min: 0, max: 18, ticks: {{ color: '#F43F5E' }}, grid: {{ display: false }} }}
            }}
          }}
        }});
      }}
    }}

    // 3. Existing Linear Trajectory & Holdout Benchmark
    function initLinearCharts() {{
      Chart.defaults.font.family = "'Plus Jakarta Sans', sans-serif";
      Chart.defaults.font.size = 11;
      Chart.defaults.color = "#8A8F98";
      const champ = getChampion();

      const trajCanvas = document.getElementById("linearTrajChart");
      if (trajCanvas) {{
        new Chart(trajCanvas.getContext("2d"), {{
          type: 'line',
          data: {{
            labels: experimentData.map(e => `#${{e.run_id}}`),
            datasets: [
              {{
                label: 'Val ROC-AUC',
                data: experimentData.map(e => e.roc),
                borderColor: '#F7F8F8',
                backgroundColor: 'rgba(255, 255, 255, 0.02)',
                borderWidth: 1.8,
                pointBackgroundColor: experimentData.map(e => e.run_id === champ.run_id ? '#4EBE9E' : '#15171A'),
                pointBorderColor: experimentData.map(e => e.run_id === champ.run_id ? '#4EBE9E' : '#8A8F98'),
                pointRadius: experimentData.map(e => e.run_id === champ.run_id ? 5 : 2.5),
                tension: 0.2,
                fill: true
              }},
              {{
                label: 'Val PR-AUC',
                data: experimentData.map(e => e.pr),
                borderColor: '#5E6AD2',
                borderWidth: 1.5,
                borderDash: [3, 3],
                pointRadius: 2,
                tension: 0.2,
                fill: false
              }}
            ]
          }},
          options: {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{
              legend: {{
                position: 'top',
                align: 'end',
                labels: {{ boxWidth: 10, font: {{ size: 10 }}, color: '#8A8F98' }}
              }}
            }},
            scales: {{
              x: {{ grid: {{ display: false }}, ticks: {{ color: '#545861' }} }},
              y: {{
                min: 0.65,
                max: 0.86,
                grid: {{ color: 'rgba(255, 255, 255, 0.04)' }},
                ticks: {{ color: '#545861' }}
              }}
            }}
          }}
        }});
      }}

      const holdoutCanvas = document.getElementById("linearHoldoutChart");
      if (holdoutCanvas) {{
        new Chart(holdoutCanvas.getContext("2d"), {{
          type: 'bar',
          data: {{
            labels: ['ROC-AUC', 'PR-AUC', 'Log Loss'],
            datasets: [
              {{
                label: '5-Fold CV (Run #12)',
                data: [0.8506, 0.6645, 0.4807],
                backgroundColor: 'rgba(255, 255, 255, 0.14)',
                borderRadius: 4,
                barPercentage: 0.55
              }},
              {{
                label: 'Sealed Holdout (1,409 rows)',
                data: [0.8443, 0.6628, 0.4736],
                backgroundColor: 'rgba(94, 106, 210, 0.4)',
                borderRadius: 4,
                barPercentage: 0.55
              }}
            ]
          }},
          options: {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{
              legend: {{
                position: 'top',
                align: 'end',
                labels: {{ boxWidth: 10, font: {{ size: 10 }}, color: '#8A8F98' }}
              }}
            }},
            scales: {{
              x: {{ grid: {{ display: false }}, ticks: {{ color: '#8A8F98' }} }},
              y: {{
                min: 0.35,
                max: 0.90,
                grid: {{ color: 'rgba(255, 255, 255, 0.04)' }},
                ticks: {{ color: '#545861' }}
              }}
            }}
          }}
        }});
      }}
    }}

    // 4. Deep-Dive Experiment Inspector (Stepper + 6 Curves)
    function buildStepper() {{
      const bar = document.getElementById('stepperBar');
      if (!bar) return;
      bar.innerHTML = '';
      runs.forEach((r, idx) => {{
        const btn = document.createElement('button');
        const isChamp = (r.experiment_id === (holdout.champion_experiment_id || 12));
        btn.className = 'step-btn' + (idx === selectedRunIdx ? ' active' : '') + (isChamp ? ' champ-step' : '');
        btn.id = 'stepBtn_' + idx;
        const famShort = (r.hyperparameters.model_family || 'cb').substring(0, 2).toUpperCase();
        btn.innerHTML = 'Exp ' + r.experiment_id + ' [' + famShort + ']' + (isChamp ? ' &#127942;' : '');
        btn.onclick = () => renderSelectedRun(idx);
        bar.appendChild(btn);
      }});
    }}

    function renderSelectedRun(idx) {{
      selectedRunIdx = idx;
      document.querySelectorAll('.step-btn').forEach((b, i) => b.classList.toggle('active', i === idx));
      const r = runs[idx];
      if (!r) return;
      const met = r.metrics || {{}};
      const hyp = r.hyperparameters || {{}};
      const famRaw = (hyp.model_family || 'catboost').toLowerCase();
      const fam = modelMap[famRaw] || hyp.model_family || 'CATBOOST';
      const isChamp = (r.experiment_id === (holdout.champion_experiment_id || 12));

      // Update Meta Tags
      const badge = document.getElementById('expBadge');
      if (badge) badge.innerHTML = `Experiment #${{r.experiment_id}}${{isChamp ? ' &bull; CHAMPION' : ''}}`;
      const famTag = document.getElementById('expFamTag');
      if (famTag) famTag.innerText = fam;

      const hypEl = document.getElementById('expHypothesis');
      if (hypEl) hypEl.innerText = r.hypothesis || 'Autonomous Iteration Step';

      const mathEl = document.getElementById('expMath');
      if (mathEl) mathEl.innerText = r.mathematical_justification || '';

      const tagsEl = document.getElementById('expConfigTags');
      if (tagsEl) {{
        tagsEl.innerHTML = `
          <span class="pill-tag">Depth: ${{hyp.depth || 6}}</span>
          <span class="pill-tag">LR: ${{hyp.learning_rate || 0.05}}</span>
          <span class="pill-tag">L2: ${{hyp.l2_leaf_reg || 3.0}}</span>
          <span class="pill-tag">Iterations: ${{hyp.iterations || 1000}}</span>
          <span class="pill-tag" style="color:var(--accent-emerald);">OOF ROC: ${{(met.oof_roc_auc || 0).toFixed(4)}}</span>
          <span class="pill-tag">OOF PR: ${{(met.oof_pr_auc || 0).toFixed(4)}}</span>
          <span class="pill-tag">Gap: ${{(met.generalization_gap || 0).toFixed(4)}}</span>
        `;
      }}

      // Destroy old chart instances
      Object.keys(expCharts).forEach(k => {{ if (expCharts[k]) expCharts[k].destroy(); }});

      // 1. Loss Trajectory
      const traj = met.loss_trajectory_samples || [];
      const ctxLoss = document.getElementById('chartExpLoss');
      if (ctxLoss) {{
        expCharts.loss = new Chart(ctxLoss.getContext('2d'), {{
          type: 'line',
          data: {{
            labels: traj.map(t => 'It ' + t.iteration),
            datasets: [
              {{ label: 'Val Loss', data: traj.map(t => t.val_loss), borderColor: '#F59E0B', tension: 0.2 }},
              {{ label: 'Train Loss', data: traj.map(t => t.train_loss), borderColor: '#38BDF8', tension: 0.2 }}
            ]
          }},
          options: {{ responsive: true, maintainAspectRatio: false, scales: {{ x: {{ ticks: {{ color: '#8A8F98' }}, grid: {{ display: false }} }}, y: {{ ticks: {{ color: '#8A8F98' }}, grid: {{ color: 'rgba(255, 255, 255, 0.04)' }} }} }} }}
        }});
      }}

      // 2. ROC Curve
      const rocPts = met.roc_curve_points || [];
      const ctxROC = document.getElementById('chartExpROC');
      if (ctxROC) {{
        expCharts.roc = new Chart(ctxROC.getContext('2d'), {{
          type: 'line',
          data: {{
            labels: rocPts.map(p => p.fpr.toFixed(2)),
            datasets: [
              {{ label: 'ROC Curve (AUC: ' + (met.oof_roc_auc || 0).toFixed(4) + ')', data: rocPts.map(p => p.tpr), borderColor: '#38BDF8', fill: true, backgroundColor: 'rgba(56, 189, 248, 0.08)' }},
              {{ label: 'Random (0.50)', data: rocPts.map(p => p.fpr), borderColor: '#545861', borderDash: [4, 4], pointRadius: 0 }}
            ]
          }},
          options: {{ responsive: true, maintainAspectRatio: false, scales: {{ x: {{ ticks: {{ color: '#8A8F98' }}, grid: {{ display: false }}, title: {{ display: true, text: 'False Positive Rate' }} }}, y: {{ ticks: {{ color: '#8A8F98' }}, grid: {{ color: 'rgba(255, 255, 255, 0.04)' }}, title: {{ display: true, text: 'True Positive Rate' }} }} }} }}
        }});
      }}

      // 3. PR Curve
      const prPts = met.pr_curve_points || [];
      const ctxPR = document.getElementById('chartExpPR');
      if (ctxPR) {{
        expCharts.pr = new Chart(ctxPR.getContext('2d'), {{
          type: 'line',
          data: {{
            labels: prPts.map(p => p.recall.toFixed(2)),
            datasets: [{{ label: 'PR Curve (AUC: ' + (met.oof_pr_auc || 0).toFixed(4) + ')', data: prPts.map(p => p.precision), borderColor: '#4EBE9E', fill: true, backgroundColor: 'rgba(78, 190, 158, 0.08)' }}]
          }},
          options: {{ responsive: true, maintainAspectRatio: false, scales: {{ x: {{ ticks: {{ color: '#8A8F98' }}, grid: {{ display: false }}, title: {{ display: true, text: 'Recall' }} }}, y: {{ ticks: {{ color: '#8A8F98' }}, grid: {{ color: 'rgba(255, 255, 255, 0.04)' }}, title: {{ display: true, text: 'Precision' }} }} }} }}
        }});
      }}

      // 4. Calibration Curve
      const calibPts = met.calibration_curve_points || [];
      const ctxCalib = document.getElementById('chartExpCalib');
      if (ctxCalib) {{
        expCharts.calib = new Chart(ctxCalib.getContext('2d'), {{
          type: 'line',
          data: {{
            labels: calibPts.map(p => p.prob_pred.toFixed(2)),
            datasets: [
              {{ label: 'Reliability (ECE: ' + (met.ece_score || 0).toFixed(4) + ')', data: calibPts.map(p => p.prob_true), borderColor: '#5E6AD2', pointRadius: 4 }},
              {{ label: 'Perfect Calibration', data: calibPts.map(p => p.prob_pred), borderColor: '#545861', borderDash: [4, 4], pointRadius: 0 }}
            ]
          }},
          options: {{ responsive: true, maintainAspectRatio: false, scales: {{ x: {{ ticks: {{ color: '#8A8F98' }}, grid: {{ display: false }}, title: {{ display: true, text: 'Mean Predicted Risk' }} }}, y: {{ ticks: {{ color: '#8A8F98' }}, grid: {{ color: 'rgba(255, 255, 255, 0.04)' }}, title: {{ display: true, text: 'Observed Fraction' }} }} }} }}
        }});
      }}

      // 5. Probability Histogram
      const histPts = met.probability_histogram || [];
      const ctxHist = document.getElementById('chartExpHist');
      if (ctxHist) {{
        expCharts.hist = new Chart(ctxHist.getContext('2d'), {{
          type: 'bar',
          data: {{
            labels: histPts.map(h => h.bin_lower.toFixed(1) + '-' + h.bin_upper.toFixed(1)),
            datasets: [{{ label: 'Predicted Accounts', data: histPts.map(h => h.count), backgroundColor: '#38BDF8', borderRadius: 4 }}]
          }},
          options: {{ responsive: true, maintainAspectRatio: false, scales: {{ x: {{ ticks: {{ color: '#8A8F98' }}, grid: {{ display: false }} }}, y: {{ ticks: {{ color: '#8A8F98' }}, grid: {{ color: 'rgba(255, 255, 255, 0.04)' }} }} }} }}
        }});
      }}

      // 6. Feature Importances
      const topF = met.top_features || [];
      const ctxFeats = document.getElementById('chartExpFeats');
      if (ctxFeats) {{
        expCharts.feats = new Chart(ctxFeats.getContext('2d'), {{
          type: 'bar',
          data: {{
            labels: topF.map(f => f.feature),
            datasets: [{{ label: 'Feature Importance (%)', data: topF.map(f => f.importance), backgroundColor: '#4EBE9E', borderRadius: 4 }}]
          }},
          options: {{ indexAxis: 'y', responsive: true, maintainAspectRatio: false, scales: {{ x: {{ ticks: {{ color: '#8A8F98' }}, grid: {{ display: false }} }}, y: {{ ticks: {{ color: '#8A8F98' }}, grid: {{ color: 'rgba(255, 255, 255, 0.04)' }} }} }} }}
        }});
      }}

      // Fold Table
      const foldHtml = (met.fold_metrics || []).map(f => `<div><b>Fold ${{f.fold}}:</b> ROC-AUC: <span style="color:var(--accent-cyan);">${{f.roc_auc.toFixed(5)}}</span> | PR-AUC: ${{f.pr_auc.toFixed(4)}} | Loss: ${{f.log_loss.toFixed(4)}}</div>`).join('');
      const foldContainer = document.getElementById('expFoldTable');
      if (foldContainer) foldContainer.innerHTML = foldHtml || 'Fold details unavailable';

      // Confusion Matrix 0.5
      const cm05 = met.confusion_matrix_05 || [[0, 0], [0, 0]];
      const cm05El = document.getElementById('expCM05');
      if (cm05El) {{
        cm05El.innerHTML = `
          <div class="cm-title">Confusion Matrix @ Default &tau; = 0.50</div>
          <div>TN: <b>${{cm05[0][0]}}</b> | FP: <b>${{cm05[0][1]}}</b></div>
          <div>FN: <b>${{cm05[1][0]}}</b> | TP: <b>${{cm05[1][1]}}</b></div>
          <div style="margin-top:4px; color:var(--accent-rose);">Missed Churners: ${{((cm05[1][0] / Math.max(1, cm05[1][0] + cm05[1][1])) * 100).toFixed(1)}}%</div>
        `;
      }}

      // Confusion Matrix Optimal
      const cmOpt = met.confusion_matrix_optimal || [[0, 0], [0, 0]];
      const optTau = met.optimal_youden_threshold !== undefined ? met.optimal_youden_threshold : 0.49;
      const cmOptEl = document.getElementById('expCMOpt');
      if (cmOptEl) {{
        cmOptEl.innerHTML = `
          <div class="cm-title">Confusion Matrix @ Optimal Youden &tau;* = ${{Number(optTau).toFixed(2)}}</div>
          <div>TN: <b>${{cmOpt[0][0]}}</b> | FP: <b>${{cmOpt[0][1]}}</b></div>
          <div>FN: <b>${{cmOpt[1][0]}}</b> | TP: <b>${{cmOpt[1][1]}}</b></div>
          <div style="margin-top:4px; color:var(--accent-emerald);">Missed Churners: ${{((cmOpt[1][0] / Math.max(1, cmOpt[1][0] + cmOpt[1][1])) * 100).toFixed(1)}}%</div>
        `;
      }}
    }}

  </script>
</body>
</html>
"""

    output_path = Path("index.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Successfully generated clean {output_path} ({len(html_content)} bytes)")

if __name__ == "__main__":
    main()
