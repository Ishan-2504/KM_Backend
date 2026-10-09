"""Structured search logging with visually formatted blocks.

Produces console output that makes it trivial to trace where relevance
breaks down for any given query.  Uses the Python logging module at INFO
level so it can be silenced in production via standard config.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

logger = logging.getLogger("km_search")

_request_counter = 0
W = 68  # box width (inner)


def _box_top() -> str:
    return f"╔{'═' * W}╗"


def _box_bottom() -> str:
    return f"╚{'═' * W}╝"


def _box_sep() -> str:
    return f"╠{'─' * W}╣"


def _box_thick_sep() -> str:
    return f"╠{'═' * W}╣"


def _line(text: str) -> str:
    return f"║ {text:<{W - 2}} ║"


def _fmt_filters(filters: dict[str, Any] | None) -> str:
    if not filters:
        return "(none)"
    parts = [f"{k}={v}" for k, v in filters.items() if v]
    return ", ".join(parts) if parts else "(none)"


@dataclass
class StepLog:
    name: str
    elapsed_ms: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchLog:
    request_id: int = 0
    raw_query: str = ""
    manual_filters: dict[str, Any] | None = None
    steps: list[StepLog] = field(default_factory=list)
    total_ms: float = 0.0
    result_count: int = 0

    def add_step(self, name: str, elapsed_ms: float, **details: Any) -> None:
        self.steps.append(StepLog(name=name, elapsed_ms=elapsed_ms, details=details))

    def emit(self) -> None:
        lines: list[str] = []
        lines.append(_box_top())
        lines.append(_line(f"KM SEARCH — Request #{self.request_id}"))
        lines.append(_box_thick_sep())
        lines.append(_line(f"RAW QUERY       : \"{self.raw_query}\""))
        lines.append(_line(f"MANUAL FILTERS  : {_fmt_filters(self.manual_filters)}"))

        for i, step in enumerate(self.steps):
            lines.append(_box_sep())
            header = f"STEP {i + 1}: {step.name}"
            ms_str = f"[{step.elapsed_ms:.1f}ms]"
            pad = W - 2 - len(header) - len(ms_str)
            lines.append(f"║ {header}{' ' * max(pad, 1)}{ms_str} ║")
            for key, val in step.details.items():
                label = key.replace("_", " ").capitalize()
                val_str = str(val)
                if len(val_str) > W - 6 - len(label):
                    val_str = val_str[: W - 9 - len(label)] + "..."
                lines.append(_line(f"  {label:<16}: {val_str}"))

        lines.append(_box_thick_sep())
        lines.append(_line(f"TOTAL LATENCY    : {self.total_ms:.1f}ms"))
        lines.append(_line(f"RESULTS RETURNED : {self.result_count}"))
        lines.append(_box_bottom())

        block = "\n".join(lines)
        logger.info("\n%s", block)


def new_search_log(query: str, manual_filters: dict[str, Any] | None = None) -> SearchLog:
    global _request_counter
    _request_counter += 1
    return SearchLog(
        request_id=_request_counter,
        raw_query=query,
        manual_filters=manual_filters,
    )


@contextmanager
def timed_step(name: str) -> Generator[dict[str, Any], None, None]:
    """Context manager that times a processing step and collects details."""
    details: dict[str, Any] = {}
    start = time.perf_counter()
    yield details
    details["__elapsed_ms"] = (time.perf_counter() - start) * 1000


def step_ms(details: dict[str, Any]) -> float:
    return details.pop("__elapsed_ms", 0.0)
