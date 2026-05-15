# Ablation study (CET-LiteFormer)

This folder holds **ablation run outputs** when you pass `--ablation_root ablation_study/runs` to the runner. Each run creates:

`ablation_study/runs/<your_experiment_name>/`

- One subdirectory per variant (checkpoints, `config_used.yaml`, `test_metrics.json`, …)
- `ablation_summary.csv` — test accuracy, macro-/weighted-F1, MCC, ROC-AUC (when defined), params, latency
- `ablation_step_deltas.csv` — gain vs the **previous step** in the same suite (incremental interpretation)
- `ablation_manifest.json` — variant names and descriptions
- `ablation_report.json` — machine-readable copy of the summary
- Plots: `ablation_macro_f1_ladder.png`, `ablation_metrics_grouped.png`, `ablation_accuracy_vs_latency.png`

## Suites

| `--suite` | What it does |
|-----------|----------------|
| `model_components` (default) | **Incremental ladder:** MLP → standard Transformer → CET (standard attn) → +correntropy → +entropy/MI gate → +learnable σ |
| `training_objectives` | Same full CET model; **loss ablation:** CE only → +focal → +exit supervision → +gate L1 |
| `legacy` | Older leave-one-out style toggles + baselines |
| `all` | Runs `model_components` then `training_objectives` (long) |

## Example (PowerShell)

From `cet_liteformer`, using CIC-Darknet2020 and writing under this folder:

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

For a **faster** dry run, copy `configs/default.yaml` to a local file, reduce `training.epochs` and `training.patience`, and pass that file to `--config`.

## Interpreting results

- **Model ladder** (`model_components`): later steps should generally improve or match earlier ones on hard traffic tasks; if a step hurts a metric, check class imbalance, variance across seeds, or whether the smaller baseline was under-trained.
- **Training ladder** (`training_objectives`): shows how focal loss, exit supervision, and gate sparsity affect the **same** architecture.
