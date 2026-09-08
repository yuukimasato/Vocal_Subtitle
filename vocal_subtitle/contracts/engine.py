"""Engine identity, availability and execution result contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from .common import CONTRACT_VERSION, ErrorInfo, jsonable


@dataclass(frozen=True)
class EngineIdentity:
    name: str
    model: str = ""
    version: str = ""
    category: str = ""
    contract_version: str = CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "version": self.version,
            "category": self.category,
            "contract_version": self.contract_version,
        }


@dataclass(frozen=True)
class EngineAvailability:
    identity: EngineIdentity
    available: bool
    status: str = "unavailable"
    reason: str = ""
    device: str = ""
    capabilities: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity.to_dict(),
            "available": self.available,
            "status": self.status,
            "reason": self.reason,
            "device": self.device,
            "capabilities": list(self.capabilities),
        }


@dataclass(frozen=True)
class EngineRequest:
    payload: Any = None
    engine: str = ""
    model: str = ""
    language: Optional[str] = None
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class PrepareResult:
    ready: bool
    identity: Optional[EngineIdentity] = None
    error: Optional[ErrorInfo] = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "identity": self.identity.to_dict() if self.identity else None,
            "error": self.error.to_dict() if self.error else None,
            "diagnostics": jsonable(self.diagnostics),
        }


@dataclass
class EngineResult:
    success: bool
    output: Any = None
    events: tuple[Any, ...] = ()
    identity: Optional[EngineIdentity] = None
    error: Optional[ErrorInfo] = None
    degraded: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "output": jsonable(self.output),
            "events": [jsonable(event) for event in self.events],
            "identity": self.identity.to_dict() if self.identity else None,
            "error": self.error.to_dict() if self.error else None,
            "degraded": self.degraded,
            "diagnostics": jsonable(self.diagnostics),
        }
