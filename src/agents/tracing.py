"""Lightweight LangChain run traces for the portfolio dashboard."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class TraceStep:
    name: str
    detail: str
    status: str = "ok"  # ok | mock | error
    started_at: str = ""
    duration_ms: float | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "detail": self.detail,
            "status": self.status,
            "started_at": self.started_at,
            "duration_ms": self.duration_ms,
            "data": self.data,
        }


@dataclass
class RunTrace:
    """One investigator (or chat) invocation, step-by-step."""

    run_id: str
    kind: str
    path_label: str
    steps: list[TraceStep] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: str | None = None

    def add(
        self,
        name: str,
        detail: str,
        *,
        status: str = "ok",
        data: dict[str, Any] | None = None,
        duration_ms: float | None = None,
    ) -> TraceStep:
        step = TraceStep(
            name=name,
            detail=detail,
            status=status,
            started_at=datetime.now(timezone.utc).isoformat(),
            duration_ms=duration_ms,
            data=data or {},
        )
        self.steps.append(step)
        return step

    def finish(self) -> None:
        self.finished_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "kind": self.kind,
            "path_label": self.path_label,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "steps": [s.to_dict() for s in self.steps],
        }


class timed_step:
    """Context manager that records duration onto a RunTrace."""

    def __init__(self, trace: RunTrace, name: str, detail: str, **data: Any):
        self.trace = trace
        self.name = name
        self.detail = detail
        self.data = data
        self._t0 = 0.0
        self.status = "ok"

    def __enter__(self) -> timed_step:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        duration = (time.perf_counter() - self._t0) * 1000
        status = "error" if exc else self.status
        detail = self.detail if not exc else f"{self.detail} · {exc}"
        self.trace.add(
            self.name,
            detail,
            status=status,
            data=self.data,
            duration_ms=round(duration, 1),
        )
