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

### Train
```powershell
cd D:\Projects\CET-LiteFormer\cet_liteformer
.\.venv\Scripts\python.exe -m src.train `
  --config configs/default.yaml `
  --csv_path "..\Datasets\CIC-Darknet2020\Darknet.CSV" `
  --label_col "Label" `
  --experiment_name "CIC-Darknet2020_CETLiteFormer" `
  --device cuda
```

### Evaluate
```powershell
cd D:\Projects\CET-LiteFormer\cet_liteformer
.\.venv\Scripts\python.exe -m src.evaluate `
  --experiment_dir "outputs\CIC-Darknet2020_CETLiteFormer" `
  --checkpoint "best_model.pt" `
  --device cuda
```

### Latency benchmark
```powershell
cd D:\Projects\CET-LiteFormer\cet_liteformer
.\.venv\Scripts\python.exe -m src.benchmark_latency `
  --experiment_dir "outputs\CIC-Darknet2020_CETLiteFormer" `
  --device cuda
```

## Dataset-specific commands (examples)

### CIC-Darknet2020 (use the “last” label)
Some CIC files include **two label columns** (pandas exposes them as `Label` and `Label.1`). After cleaning, the second one becomes `Label__dup1`.

```powershell
cd D:\Projects\CET-LiteFormer\cet_liteformer
.\.venv\Scripts\python.exe -m src.train `
  --config configs/default.yaml `
  --csv_path "..\Datasets\CIC-Darknet2020\Darknet.CSV" `
  --label_col "Label__dup1" `
  --experiment_name "CIC-Darknet2020_CETLiteFormer_LastLabel" `
  --device cuda
```

### BCCC-Darknet-2025 (Binary)
This dataset uses a lowercase label column: **`label`**.

```powershell
cd D:\Projects\CET-LiteFormer\cet_liteformer
.\.venv\Scripts\python.exe -m src.train `
  --config configs/default.yaml `
  --csv_path "..\Datasets\BCCC-Darknet-2025\Binary -2DSCombined.csv" `
  --label_col "label" `
  --experiment_name "BCCC-Darknet-2025_Binary_CETLiteFormer" `
  --device cuda
```

### BCCC-Darknet-2025 (Multi-class)
```powershell
cd D:\Projects\CET-LiteFormer\cet_liteformer
.\.venv\Scripts\python.exe -m src.train `
  --config configs/default.yaml `
  --csv_path "..\Datasets\BCCC-Darknet-2025\MultiTotalDS.csv" `
  --label_col "label" `
  --experiment_name "BCCC-Darknet-2025_Multi_CETLiteFormer" `
  --device cuda
```

### ISCX-Tor-NonTor-2017 (Scenario A / B)
```powershell
cd D:\Projects\CET-LiteFormer\cet_liteformer
.\.venv\Scripts\python.exe -m src.train `
  --config configs/default.yaml `
  --csv_path "..\Datasets\ISCX-Tor-NonTor-2017\Scenario-A-merged_5s.csv" `
  --label_col "label" `
  --experiment_name "iscx_tor_scenario_a_cet_liteformer" `
  --device cuda
```

```powershell
cd D:\Projects\CET-LiteFormer\cet_liteformer
.\.venv\Scripts\python.exe -m src.train `
  --config configs/default.yaml `
  --csv_path "..\Datasets\ISCX-Tor-NonTor-2017\Scenario-B-merged_5s.csv" `
  --label_col "label" `
  --experiment_name "iscx_tor_scenario_b_cet_liteformer" `
  --device cuda
```

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
- **Label column not found**: print/inspect headers and pass the exact name via `--label_col` (common: `label`, `Label`, `Label__dup1`).
- **CUDA not used**: ensure your PyTorch build is CUDA-enabled and run with `--device cuda`.

## License
Research / academic use. Add a LICENSE file if you plan to publish as open source.
