#!/usr/bin/env bash
set -e

python -m src.train \
  --config configs/default.yaml \
  --csv_path "../Datasets/CIC-Darknet2020/Darknet.CSV" \
  --label_col "Label" \
  --experiment_name "CIC-Darknet2020_CETLiteFormer-tor-vpn"

python -m src.evaluate \
  --experiment_dir "outputs/CIC-Darknet2020_CETLiteFormer-tor-vpn"

python -m src.benchmark_latency \
  --experiment_dir "outputs/CIC-Darknet2020_CETLiteFormer-tor-vpn"
