from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from ..utils.io import ensure_dir, save_json


GROUP_KEYWORDS: Dict[str, List[str]] = {
    "duration": ["duration"],
    "packet_length": ["len", "length", "packet length", "pkt len"],
    "packet_count": ["tot fwd pkts", "tot bwd pkts", "packet", "pkt", "packets_count", "fwd_packets", "bwd_packets"],
    "byte_count": ["byte", "bytes", "payload", "header_bytes", "total_payload_bytes", "total_header_bytes"],
    "iat": ["iat", "inter arrival", "delta_time"],
    "flow_rate": ["flow bytes/s", "flow packets/s", "rate", "/s", "bytes_rate", "packets_rate"],
    "flags": ["flag", "syn", "ack", "fin", "rst", "psh", "urg", "ece", "cwr"],
    "header": ["header"],
    "active_idle": ["active", "idle"],
    "protocol_port": ["protocol", "port"],
    "statistical": ["mean", "std", "var", "variance", "min", "max", "avg", "median", "skew", "cov", "mode"],
    "other": [],
}


def build_feature_groups(feature_names: Sequence[str]) -> Tuple[List[int], List[str], Dict[str, int]]:
    """
    Assign each feature to a group based on name matching (lowercase substring).

    Returns:
      - group_ids: List[int] per feature index
      - group_names: canonical group name per feature index
      - group_to_id: mapping of group name to id
    """
    groups_order = [
        "duration",
        "packet_length",
        "packet_count",
        "byte_count",
        "iat",
        "flow_rate",
        "flags",
        "header",
        "active_idle",
        "protocol_port",
        "statistical",
        "other",
    ]
    group_to_id = {g: i for i, g in enumerate(groups_order)}

    group_ids: List[int] = []
    group_names: List[str] = []
    for name in feature_names:
        s = str(name).strip().lower()
        assigned = "other"
        for g in groups_order:
            if g == "other":
                continue
            for kw in GROUP_KEYWORDS.get(g, []):
                if kw in s:
                    assigned = g
                    break
            if assigned != "other":
                break
        group_ids.append(group_to_id[assigned])
        group_names.append(assigned)

    return group_ids, group_names, group_to_id


def save_feature_groups(
    output_dir: str | Path,
    feature_names: Sequence[str],
    group_ids: Sequence[int],
    group_names: Sequence[str],
    group_to_id: Dict[str, int],
) -> Path:
    out_dir = Path(output_dir)
    ensure_dir(out_dir)
    path = out_dir / "feature_groups.json"
    payload = {
        "group_to_id": group_to_id,
        "features": [
            {"feature_name": str(n), "group_id": int(gid), "group_name": str(gn)}
            for n, gid, gn in zip(feature_names, group_ids, group_names)
        ],
    }
    save_json(payload, path)
    return path

