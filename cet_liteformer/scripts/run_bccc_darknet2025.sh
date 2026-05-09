#!/usr/bin/env bash
set -e

python -m src.train \
  --config configs/default.yaml \
  --csv_path "../Datasets/BCCC-Darknet-2025/MultiTotalDS.csv" \
  --label_col "label" \
  --experiment_name "bccc_darknet2025_multitotal_cet_liteformer"

python -m src.evaluate \
  --experiment_dir "outputs/bccc_darknet2025_multitotal_cet_liteformer"

python -m src.benchmark_latency \
  --experiment_dir "outputs/bccc_darknet2025_multitotal_cet_liteformer"
