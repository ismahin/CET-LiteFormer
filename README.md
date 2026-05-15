# CET-LiteFormer

**CET-LiteFormer** is a PyTorch implementation of a lightweight Transformer for **flow-level darknet traffic classification**, with a practical **CSV/ARFF preprocessing pipeline**, end-to-end training, evaluation (train/val/test), and latency benchmarking.

## What you get
- **End-to-end pipeline**: load → clean → split → scale (train-only) → train → evaluate → export artifacts to `outputs/`.
- **Feature grouping + MI prior**: saves `feature_groups.json` and `feature_mi_scores.csv` for reproducible checkpoints.
- **Correlation feature selection (train-only)**: optional Spearman selection and plots.
- **Evaluation artifacts**: metrics JSON, classification report CSV, confusion matrix, ROC curve, predictions.
- **Latency benchmarking**: throughput/latency reporting and early-exit comparisons.

## Requirements
- **Windows 10/11** (PowerShell commands below) or Linux/macOS (bash is similar).
- **Python 3.10+** (your setup uses Python 3.12).
- **PyTorch** with CUDA if you want GPU training (e.g. `cu121` builds).

## Setup (Windows / PowerShell)

```powershell
cd D:\Projects\CET-LiteFormer\cet_liteformer
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Quickstart (commands)

This repository keeps **three** public benchmark setups (see below). Artifacts for those runs live under `cet_liteformer/outputs/<experiment_name>/`.

### Train (example: CIC-Darknet2020, first label column)
```powershell
cd D:\Projects\CET-LiteFormer\cet_liteformer
.\.venv\Scripts\python.exe -m src.train `
  --config configs/default.yaml `
  --csv_path "..\Datasets\CIC-Darknet2020\Darknet.CSV" `
  --label_col "Label" `
  --experiment_name "CIC-Darknet2020_CETLiteFormer-tor-vpn" `
  --device cuda
```

### Evaluate
```powershell
cd D:\Projects\CET-LiteFormer\cet_liteformer
.\.venv\Scripts\python.exe -m src.evaluate `
  --experiment_dir "outputs\CIC-Darknet2020_CETLiteFormer-tor-vpn" `
  --checkpoint "best_model.pt" `
  --device cuda
```

### Latency benchmark
```powershell
cd D:\Projects\CET-LiteFormer\cet_liteformer
.\.venv\Scripts\python.exe -m src.benchmark_latency `
  --experiment_dir "outputs\CIC-Darknet2020_CETLiteFormer-tor-vpn" `
  --device cuda
```

## Dataset-specific commands (three benchmarks)

### CIC-Darknet2020 (first `Label` column)
Some CIC files include **two label columns** (pandas may show `Label` and `Label.1`). This setup uses the first: **`Label`**. The checked-in run folder is `outputs\CIC-Darknet2020_CETLiteFormer-tor-vpn\`.

```powershell
cd D:\Projects\CET-LiteFormer\cet_liteformer
.\.venv\Scripts\python.exe -m src.train `
  --config configs/default.yaml `
  --csv_path "..\Datasets\CIC-Darknet2020\Darknet.CSV" `
  --label_col "Label" `
  --experiment_name "CIC-Darknet2020_CETLiteFormer-tor-vpn" `
  --device cuda
```

### ISCX-Tor-NonTor-2017 (Scenario A)
```powershell
cd D:\Projects\CET-LiteFormer\cet_liteformer
.\.venv\Scripts\python.exe -m src.train `
  --config configs/default.yaml `
  --csv_path "..\Datasets\ISCX-Tor-NonTor-2017\Scenario-A-merged_5s.csv" `
  --label_col "label" `
  --experiment_name "iscx_tor_scenario_a_cet_liteformer" `
  --device cuda
```

### ISCXVPN2016 (Scenario A1, 15s VPN ARFF)
```powershell
cd D:\Projects\CET-LiteFormer\cet_liteformer
.\.venv\Scripts\python.exe -m src.train `
  --config configs/default.yaml `
  --csv_path "..\Datasets\ISCXVPN2016  VPN-nonVPN\Scenario A1-ARFF\TimeBasedFeatures-Dataset-15s-VPN.arff" `
  --label_col "label" `
  --experiment_name "iscxvpn2016_a1_15s_vpn_cet_liteformer" `
  --device cuda
```

Bash equivalents for train/eval are under `cet_liteformer/scripts/`.

### Evaluate (test metrics; all three benchmarks)

**CIC-Darknet2020**

```powershell
cd D:\Projects\CET-LiteFormer\cet_liteformer
.\.venv\Scripts\python.exe -m src.evaluate `
  --experiment_dir "outputs\CIC-Darknet2020_CETLiteFormer-tor-vpn" `
  --checkpoint "best_model.pt" `
  --device cuda
```

**ISCX-Tor Scenario A**

```powershell
cd D:\Projects\CET-LiteFormer\cet_liteformer
.\.venv\Scripts\python.exe -m src.evaluate `
  --experiment_dir "outputs\iscx_tor_scenario_a_cet_liteformer" `
  --checkpoint "best_model.pt" `
  --device cuda
```

**ISCXVPN2016 Scenario A1**

```powershell
cd D:\Projects\CET-LiteFormer\cet_liteformer
.\.venv\Scripts\python.exe -m src.evaluate `
  --experiment_dir "outputs\iscxvpn2016_a1_15s_vpn_cet_liteformer" `
  --checkpoint "best_model.pt" `
  --device cuda
```

Optional: `--checkpoint last_model.pt`, `--device cpu`, `--batch_size 256`.

### Latency benchmark (all three)

**CIC-Darknet2020**

```powershell
cd D:\Projects\CET-LiteFormer\cet_liteformer
.\.venv\Scripts\python.exe -m src.benchmark_latency `
  --experiment_dir "outputs\CIC-Darknet2020_CETLiteFormer-tor-vpn" `
  --device cuda
```

**ISCX-Tor Scenario A**

```powershell
cd D:\Projects\CET-LiteFormer\cet_liteformer
.\.venv\Scripts\python.exe -m src.benchmark_latency `
  --experiment_dir "outputs\iscx_tor_scenario_a_cet_liteformer" `
  --device cuda
```

**ISCXVPN2016 Scenario A1**

```powershell
cd D:\Projects\CET-LiteFormer\cet_liteformer
.\.venv\Scripts\python.exe -m src.benchmark_latency `
  --experiment_dir "outputs\iscxvpn2016_a1_15s_vpn_cet_liteformer" `
  --device cuda
```

## Ablation study

Runs multiple model or training variants and writes summaries under `cet_liteformer/ablation_study/runs/<experiment_name>/` (when using `--ablation_root` as below). See `cet_liteformer/ablation_study/README.md` for suite details (`model_components`, `training_objectives`, `legacy`, `all`).

**Example: incremental model ladder on CIC-Darknet2020**

```powershell
cd D:\Projects\CET-LiteFormer\cet_liteformer
.\.venv\Scripts\python.exe -m src.run_ablation `
  --config configs/default.yaml `
  --csv_path "..\Datasets\CIC-Darknet2020\Darknet.CSV" `
  --label_col "Label" `
  --experiment_name "cic_darknet2020_components" `
  --suite model_components `
  --ablation_root "ablation_study/runs" `
  --device cuda
```

**Example: training-objective ablations (same full CET model)**

```powershell
cd D:\Projects\CET-LiteFormer\cet_liteformer
.\.venv\Scripts\python.exe -m src.run_ablation `
  --config configs/default.yaml `
  --csv_path "..\Datasets\CIC-Darknet2020\Darknet.CSV" `
  --label_col "Label" `
  --experiment_name "cic_darknet2020_training_obj" `
  --suite training_objectives `
  --ablation_root "ablation_study/runs" `
  --device cuda
```

For faster iteration, point `--config` at a YAML copy with reduced `training.epochs` and `training.patience`.

## Configuration
The main config is `configs/default.yaml`. Useful knobs:
- **Data**: split ratios, scaling, correlation feature selection (`data.correlation_selection`).
- **Training**: epochs, batch size, optimizer, early stopping (`training.patience`).
- **Evaluation**: set `evaluation.measure_latency: true` to write latency stats during `src.evaluate`.

## Outputs
Each run writes to `outputs/<experiment_name>/` (names may vary slightly by config):
- **Checkpoints**: `best_model.pt`, `last_model.pt`
- **Repro artifacts**: `config_used.yaml`, `scaler.joblib`, `label_mapping.json`, `feature_metadata.json`
- **Feature metadata**: `feature_mi_scores.csv`, `feature_groups.json` (and correlation artifacts when enabled)
- **Training log**: `training_log.csv` (includes `train_loss` and `val_loss`)
- **Evaluation**: per-split metrics JSON/CSV, `classification_report.csv`, predictions CSV, confusion matrix (`.png`/`.csv`), ROC curve (`roc_curve.png` when applicable)
- **Latency**: `latency_report.json` / `latency_report.csv` from `src.benchmark_latency`

## Troubleshooting
- **PowerShell**: use `;` instead of `&&` when chaining commands.
- **Label column not found**: print/inspect headers and pass the exact name via `--label_col` (common: `label`, `Label`; a duplicate second column may appear as `Label__dup1`).
- **CUDA not used**: ensure your PyTorch build is CUDA-enabled and run with `--device cuda`.

## License
Research / academic use. Add a LICENSE file if you plan to publish as open source.
