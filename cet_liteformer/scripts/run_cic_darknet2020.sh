#!/usr/bin/env bash
set -e

python -m src.train \
  --config configs/default.yaml \
  --csv_path "../Datasets/CIC-Darknet2020/Darknet.CSV" \
  --label_col "Label" \
  --experiment_name "cic_darknet2020_cet_liteformer"

python -m src.evaluate \
  --experiment_dir "outputs/cic_darknet2020_cet_liteformer"

python -m src.benchmark_latency \
  --experiment_dir "outputs/cic_darknet2020_cet_liteformer"
