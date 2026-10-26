# Google Colab Training Guide

This guide details how to execute the **Autonomous AI Data Scientist (AADS)** framework inside **Google Colab** with hardware acceleration (NVIDIA GPU), seamless secrets management, and automated artifact persistence.

---

## 1. Quickstart: One-Click Execution

Click the badge below to open the official interactive training notebook directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uday0214/IOTAxEngimaHackathon/blob/main/notebooks/colab_training.ipynb)

---

## 2. Hardware Accelerator Configuration

CatBoost model training is heavily accelerated by CUDA-enabled NVIDIA GPUs (up to 4x–8x speedup over CPU for 5-fold cross-validation across 15 iterations).

### Recommended Colab Runtime Setup
1. In the Google Colab top navigation bar, select **Runtime** $\rightarrow$ **Change runtime type**.
2. **Hardware accelerator**: Select **T4 GPU** (available on free tier) or **A100 / V100 GPU** (Colab Pro).
3. **Python runtime**: Python 3.
4. Click **Save**.
5. Verify GPU availability in a code cell:
   ```bash
   !nvidia-smi
   ```

> [!NOTE]
> The framework includes an automatic hardware abstraction layer (`src/utils/environment.py`). If you execute on a standard CPU runtime or if CUDA drivers are temporarily unavailable, the framework will detect this and automatically fall back to CPU execution without crashing.

---

## 3. Gemini API Key & Colab Secrets Setup

The **Gemini Supervisor Agent** queries Google AI Studio's `gemini-2.5-flash` model to analyze telemetry and generate the scientific hypothesis for each experiment.

### Secure Key Injection via Colab Secrets (Recommended)
You do **not** need to hardcode API keys or commit `.env` files into Google Colab. The framework natively integrates with Google Colab's built-in **Secrets Manager**:

1. In the left sidebar of Google Colab, click the **Secrets (Key icon)** tab.
2. Click **+ Add new secret**.
3. Set the Name to: `GEMINI_API_KEY`
4. Set the Value to: Your Gemini API key from [Google AI Studio](https://aistudio.google.com/).
5. Toggle **Notebook access** to **ON**.

```python
# The framework automatically checks google.colab.userdata:
from src.utils.environment import get_gemini_api_key

api_key = get_gemini_api_key()
# Automatically resolves from Colab Secrets!
```

> [!TIP]
> If no API key is configured, the system automatically falls back to **MOCK_SUPERVISOR** mode. In mock mode, the full 15-experiment loop, data provenance guards, Optuna bridge, and CatBoost fitting run deterministically without external API calls.

---

## 4. End-to-End Workflow in Google Colab

### Step 1: Clone Repository & Install Dependencies
```bash
!git clone https://github.com/uday0214/IOTAxEngimaHackathon.git
%cd IOTAxEngimaHackathon
!pip install -q -r requirements.txt
```

### Step 2: Data Ingestion & Cryptographic Manifest Generation
Prepare the canonical 80% train-val and 20% sealed holdout datasets and generate the immutable SHA-256 fingerprint registry:
```bash
!python scripts/prepare_data.py
```
*Zero experiment budget cost: data preparation, cleaning, and partitioning do not count towards the 15-model-fit invariant.*

### Step 3: Execute the 15-Run Autonomous Training Loop
Run the live autonomous optimization loop with GPU acceleration:
```bash
!python scripts/run_autonomous_loop.py --live --device auto --clean-ledger
```

#### CLI Parameters:
- `--live`: Enables actual CatBoost 5-fold cross-validation fitting (omitting `--live` runs simulation mode for rapid pipeline testing).
- `--device {auto, gpu, cpu}`:
  - `auto` (default): Detects CUDA GPU and selects GPU acceleration if available; otherwise falls back to CPU.
  - `gpu`: Enforces CUDA acceleration for CatBoost (`task_type="GPU"`).
  - `cpu`: Fits on CPU cores.
- `--clean-ledger`: Resets previous experiment records in `artifacts/experiment_ledger.jsonl`.
- `--max-runs 15`: Strictly enforces the 15-experiment budget ceiling.
- `--colab`: Enables Google Drive auto-mounting for persistence.

### Step 4: Evaluate Sealed Holdout & Export Competition Submission
After Experiment 15 concludes, evaluate the champion model on the sealed holdout set and generate the official submission file:
```bash
!python scripts/evaluate_holdout.py
!python scripts/export_submission.py
```

Outputs generated:
- `artifacts/champion_model.cbm`: Serialized champion CatBoost model weights.
- `artifacts/holdout_evaluation.json`: Sealed holdout ROC-AUC, PR-AUC, and log-loss evaluation.
- `artifacts/submission.csv`: Standardized predictions matching organizer schema.
- `artifacts/experiment_ledger.jsonl`: Complete audit trail of all 15 experiments.

---

## 5. Visualizing Telemetry in Colab

Inside the notebook, you can interactively visualize loss convergence, metric progression, and hyperparameter trajectories:

```python
import json
import pandas as pd
import matplotlib.pyplot as plt

with open("artifacts/experiment_ledger.jsonl", "r") as f:
    runs = [json.loads(line) for line in f if line.strip()]

df = pd.DataFrame([{
    "Run": r["experiment_id"],
    "ROC_AUC": r["metrics"]["oof_roc_auc"],
    "PR_AUC": r["metrics"]["oof_pr_auc"],
    "Val_Loss": r["metrics"]["final_val_loss"],
} for r in runs])

fig, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(df["Run"], df["ROC_AUC"], marker="o", color="#1f77b4", label="OOF ROC-AUC")
ax1.plot(df["Run"], df["PR_AUC"], marker="s", color="#2ca02c", label="OOF PR-AUC")
ax1.set_ylabel("AUC Score")
ax1.set_xlabel("Experiment Run")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(loc="upper left")
plt.title("Autonomous Optimization Trajectory (15 Experiments)")
plt.show()
```

---

## 6. Persisting Artifacts to Google Drive

To ensure experiment results and champion model weights persist across Colab session disconnections:

```python
from google.colab import drive
import shutil

# Mount Google Drive
drive.mount('/content/drive')

# Copy artifacts to Google Drive
drive_dest = "/content/drive/MyDrive/AADS_Telco_Artifacts"
shutil.copytree("artifacts", drive_dest, dirs_exist_ok=True)
print(f"Artifacts successfully backed up to {drive_dest}")
```

Or download the archived artifacts directly through your browser:
```python
from google.colab import files
import shutil

shutil.make_archive("aads_artifacts", "zip", "artifacts")
files.download("aads_artifacts.zip")
files.download("artifacts/submission.csv")
```

---

## 7. Troubleshooting in Google Colab

| Issue | Cause | Resolution |
| :--- | :--- | :--- |
| `CatBoostError: CUDA driver version is insufficient` | Non-GPU runtime or mismatched CUDA driver | The framework automatically falls back to CPU. To use GPU, switch Colab runtime to **T4 GPU** under **Runtime -> Change runtime type**. |
| `KeyError: GEMINI_API_KEY` | Colab Secret not toggled or missing | Ensure you checked **Notebook access** to ON in the Colab Secrets tab, or set `export GEMINI_API_KEY=...`. The system will default to Mock mode if absent. |
| Session Disconnection / Timeout | Colab 12-hour timeout or idle timeout | The append-only ledger `artifacts/experiment_ledger.jsonl` saves every single run after completion. Training can resume without losing prior progress. |
| Out of Memory (OOM) | Memory spike from large depth or iterations | Default search space constrains depth $\le 8$ and iterations $\le 2500$, fitting comfortably within Colab's 12.7 GB system RAM and 15 GB GPU VRAM. |
