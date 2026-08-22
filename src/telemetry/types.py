from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SCHEMA_VERSION = "telemetry_record.v1"


@dataclass(frozen=True)
class TelemetrySource:
    """Transport-agnostic description of where a telemetry record came from."""

    type: str
    name: str | None = None
    received_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.type.strip():
            raise ValueError("TelemetrySource.type is required.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "name": self.name,
            "received_at": self.received_at,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class TelemetryRecord:
    """Canonical internal telemetry observation independent of transport."""

    telemetry: dict[str, Any]
    source: TelemetrySource
    schema_version: str = SCHEMA_VERSION
    timestamp: str | None = None
    sequence_id: int | None = None
    mission_id: str | None = None
    spacecraft_id: str | None = None
    subsystem: str | None = None
    context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.schema_version.strip():
            raise ValueError("TelemetryRecord.schema_version is required.")
        if not self.telemetry:
            raise ValueError("TelemetryRecord.telemetry must contain at least one field.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "mission_id": self.mission_id,
            "spacecraft_id": self.spacecraft_id,
            "timestamp": self.timestamp,
            "sequence_id": self.sequence_id,
            "subsystem": self.subsystem,
            "source": self.source.to_dict(),
            "telemetry": self.telemetry,
            "context": self.context,
            "metadata": {"schema_version": self.schema_version, **self.metadata},
        }

    @property
    def record_id(self) -> str | None:
        value = self.metadata.get("record_id")
        return str(value) if value is not None else None
