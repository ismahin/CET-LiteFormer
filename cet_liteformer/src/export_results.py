from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .utils.io import load_json


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs_dir", type=str, default="outputs")
    ap.add_argument("--out_csv", type=str, default="outputs/experiment_summary.csv")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.outputs_dir)
    rows: List[Dict[str, Any]] = []
    for exp in out_dir.glob("*"):
        if not exp.is_dir():
            continue
        metrics_path = exp / "test_metrics.json"
        if not metrics_path.exists():
            continue
        metrics = load_json(metrics_path)
        row = {"experiment": exp.name, **metrics}
        lat_path = exp / "latency_report.json"
        if lat_path.exists():
            lat = load_json(lat_path)
            # include only the headline latency if present
            try:
                row["latency_ms_bs1_early_exit_disabled"] = lat["early_exit_disabled"]["latency_ms_bs1"]
                row["latency_ms_bs1_early_exit_enabled"] = lat["early_exit_enabled"]["latency_ms_bs1"]
            except Exception:
                pass
        rows.append(row)

    df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()

