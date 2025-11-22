"""Console logger utilities for the TRELLIS helper scripts."""
from __future__ import annotations

import datetime as _dt
import os
import platform
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

_INDENT = "    "
_MIN_WIDTH = 60
_MAX_WIDTH = 120


class ConsoleLogger:
    """Styled console logger that prints to stdout and logs to disk."""

    def __init__(self, name: str, width: int = 80) -> None:
        self.name = name
        self.width = max(_MIN_WIDTH, min(width, _MAX_WIDTH))
        timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(__file__).parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = log_dir / f"{name}_{timestamp}.log"
        self._write_file(f"[{timestamp}] {name} log started\n")

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------
    def header(self, title: str) -> None:
        bar = "=" * self.width
        self._emit(bar)
        self._emit(title.center(self.width))
        self._emit(bar)

    def divider(self, char: str = "-") -> None:
        self._emit(char * self.width)

    def instructions(self, lines: Sequence[str]) -> None:
        self._emit("Instructions:")
        for idx, line in enumerate(lines, start=1):
            self._emit(f"{_INDENT}{idx}. {line}")

    def env_info(self) -> None:
        self.section("Environment", icon="ℹ")
        info = {
            "Platform": platform.platform(),
            "Python": sys.version.split()[0],
            "Executable": sys.executable,
            "Working dir": os.getcwd(),
        }
        for key, value in info.items():
            self._emit(f"{_INDENT}{key}: {value}")

    def section(self, title: str, icon: Optional[str] = None) -> None:
        prefix = f"{icon} " if icon else ""
        self.divider()
        self._emit(f"{prefix}{title}")
        self.divider()

    def subsection(self, title: str) -> None:
        self._emit(title)
        self._emit("-" * len(title))

    def step(self, index: int, total: int, message: str) -> None:
        self._emit(f"[{index}/{max(total, 1)}] {message}")

    def box(self, title: Optional[str], lines: Sequence[str]) -> None:
        content = list(lines)
        inner_width = max((len(line) for line in content), default=0)
        box_width = min(self.width, max(inner_width + 4, _MIN_WIDTH))
        horizontal = "+" + "-" * (box_width - 2) + "+"
        self._emit(horizontal)
        if title:
            title_str = title if isinstance(title, str) else str(title)
            self._emit(f"| {title_str.center(box_width - 4)} |")
            self._emit(horizontal)
        for line in content:
            padded = line.ljust(box_width - 4)
            self._emit(f"| {padded} |")
        self._emit(horizontal)

    def summary(self, success: bool, message: str) -> None:
        status = "SUCCESS" if success else "FAILED"
        self.box(title="Summary", lines=[f"Status: {status}", message])

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def plain(self, text: str, indent: int = 0) -> None:
        self._emit(text, indent)

    def info(self, text: str, indent: int = 0) -> None:
        self._emit(f"[INFO] {text}", indent)

    def success(self, text: str, indent: int = 0) -> None:
        self._emit(f"[OK] {text}", indent)

    def warning(self, text: str, indent: int = 0) -> None:
        self._emit(f"[WARN] {text}", indent)

    def error(self, text: str, indent: int = 0) -> None:
        self._emit(f"[ERROR] {text}", indent)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _emit(self, text: str, indent: int = 0) -> None:
        message = f"{_INDENT * max(indent, 0)}{text}"
        print(message)
        self._write_file(message + "\n")

    def _write_file(self, text: str) -> None:
        with self.log_file.open("a", encoding="utf-8") as handle:
            handle.write(text)
