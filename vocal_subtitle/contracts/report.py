"""Versioned run-report contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .common import CONTRACT_VERSION, REPORT_SCHEMA_VERSION, jsonable


@dataclass
class RunReport:
    run_id: str
    task_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    report_schema_version: str = REPORT_SCHEMA_VERSION
    contract_version: str = CONTRACT_VERSION
    _legacy_report: Any = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        result = dict(jsonable(self.payload))
        result.setdefault("run_id", self.run_id)
        result.setdefault("task_id", self.task_id)
        result.setdefault("$schema", self.report_schema_version)
        result.setdefault("contract_version", self.contract_version)
        return result

    @classmethod
    def from_legacy(cls, report: Any) -> RunReport:
        payload = report.to_dict() if hasattr(report, "to_dict") else dict(report or {})
        return cls(
            run_id=str(payload.get("run_id", getattr(report, "run_id", ""))),
            task_id=str(payload.get("task_id", getattr(report, "task_id", ""))),
            payload=payload,
            report_schema_version=str(payload.get("$schema", REPORT_SCHEMA_VERSION)),
            _legacy_report=report,
        )
