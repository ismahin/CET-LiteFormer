from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .io import ensure_dir


def print_section(title: str) -> None:
    # Windows consoles can fail to print box-drawing characters depending on codepage.
    # Prefer the requested style, but fall back to ASCII separators if needed.
    sep = "────────────────────────────────────────"
    try:
        print(sep)
        print(title)
        print(sep)
    except UnicodeEncodeError:
        sep2 = "-" * 40
        print(sep2)
        print(title)
        print(sep2)


def now_str() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class CSVLogger:
    def __init__(self, path: str | Path, fieldnames: Iterable[str]) -> None:
        self.path = Path(path)
        ensure_dir(self.path.parent)
        self.fieldnames = list(fieldnames)
        self._initialized = False

    def _init(self) -> None:
        if self._initialized:
            return
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
        self._initialized = True

    def log(self, row: Dict[str, Any]) -> None:
        self._init()
        safe_row = {k: row.get(k, None) for k in self.fieldnames}
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(safe_row)

