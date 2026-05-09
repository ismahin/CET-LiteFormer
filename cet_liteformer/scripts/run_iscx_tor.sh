#!/usr/bin/env bash
set -e

python -m src.train \
  --config configs/default.yaml \
  --csv_path "../Datasets/ISCX-Tor-NonTor-2017/Scenario-A-merged_5s.csv" \
  --label_col "label" \
  --experiment_name "iscx_tor_scenario_a_cet_liteformer"

python -m src.evaluate \
  --experiment_dir "outputs/iscx_tor_scenario_a_cet_liteformer"

python -m src.benchmark_latency \
  --experiment_dir "outputs/iscx_tor_scenario_a_cet_liteformer"
