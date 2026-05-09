#!/usr/bin/env bash
set -e

# Example ARFF file from ISCXVPN2016
python -m src.train \
  --config configs/default.yaml \
  --csv_path "../Datasets/ISCXVPN2016  VPN-nonVPN/Scenario A1-ARFF/TimeBasedFeatures-Dataset-15s-VPN.arff" \
  --label_col "label" \
  --experiment_name "iscxvpn2016_a1_15s_vpn_cet_liteformer"

python -m src.evaluate \
  --experiment_dir "outputs/iscxvpn2016_a1_15s_vpn_cet_liteformer"

python -m src.benchmark_latency \
  --experiment_dir "outputs/iscxvpn2016_a1_15s_vpn_cet_liteformer"
