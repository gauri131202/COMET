from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DictionaryEntry:
    series_key: str
    name: str | None
    dictionary_type: str | None
    description: str | None
    units_eu: str | None
    units_dn: str | None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TelemetryDictionary:
    entries: dict[str, DictionaryEntry]
    # Alarm definitions for channel ids that have no entry in the channel file.
    orphan_alarms: dict[str, list[dict[str, str]]] = field(default_factory=dict)


def read_telemetry_dictionary(
    channels_path: str | Path, alarms_path: str | Path | None = None
) -> TelemetryDictionary:
    """Read a channel dictionary (and optionally its alarms) keyed by channel abbreviation."""
    root = _parse_xml(channels_path)
    enums = _read_enum_tables(root)

    entries: dict[str, DictionaryEntry] = {}
    for element in root.iter("telemetry"):
        key = element.get("abbreviation")
        if not key:
            raise ValueError(f"{channels_path}: <telemetry> without an abbreviation")
        if key in entries:
            raise ValueError(f"{channels_path}: duplicate channel abbreviation {key!r}")
        entries[key] = _read_entry(element, key, enums)

    orphans: dict[str, list[dict[str, str]]] = {}
    if alarms_path is not None:
        for key, alarms in _read_alarms(alarms_path).items():
            if key in entries:
                extra = dict(entries[key].extra)
                extra["alarms"] = alarms
                entries[key] = _replace_extra(entries[key], extra)
            else:
                orphans[key] = alarms

    return TelemetryDictionary(entries=entries, orphan_alarms=orphans)


def _parse_xml(path: str | Path) -> ET.Element:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Dictionary file not found: {path}")
    try:
        return ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"{path}: invalid XML ({exc})") from exc


def _read_enum_tables(root: ET.Element) -> dict[str, list[dict[str, str | None]]]:
    tables: dict[str, list[dict[str, str | None]]] = {}
    for table in root.iter("enum_table"):
        name = table.get("name")
        if name:
            tables[name] = [
                {"symbol": e.get("symbol"), "numeric": e.get("numeric")} for e in table.iter("enum")
            ]
    return tables


def _read_entry(
    element: ET.Element, key: str, enums: dict[str, list[dict[str, str | None]]]
) -> DictionaryEntry:
    extra: dict[str, Any] = {
        "source": element.get("source"),
        "byte_length": element.get("byte_length"),
        "measurement_id": _text(element, "measurement_id"),
        "format": _text(element, "format"),
    }

    enum_format = element.find("enum_format")
    if enum_format is not None:
        enum_name = enum_format.get("enum_name")
        extra["enum_name"] = enum_name
        extra["enum_values"] = enums.get(enum_name or "", [])

    units_eu = None
    raw_to_eng = element.find("raw_to_eng")
    if raw_to_eng is not None:
        units_eu = _text(raw_to_eng, "eng_units")
        extra["raw_to_eng"] = [
            {"index": f.get("index"), "coeff": f.get("coeff")}
            for f in raw_to_eng.iter("factor")
        ]

    return DictionaryEntry(
        series_key=key,
        name=element.get("name"),
        dictionary_type=element.get("type"),
        description=_text(element, "description"),
        units_eu=units_eu,
        units_dn=_text(element, "raw_units"),
        extra={k: v for k, v in extra.items() if v is not None},
    )


def _read_alarms(path: str | Path) -> dict[str, list[dict[str, str]]]:
    root = _parse_xml(path)
    alarms: dict[str, list[dict[str, str]]] = {}
    for alarm in root.iter("alarm"):
        key = alarm.get("channel_id")
        if not key:
            raise ValueError(f"{path}: <alarm> without a channel_id")
        for rule in alarm:
            alarms.setdefault(key, []).append(
                {"test_type": alarm.get("test_type") or "", "rule": rule.tag, **rule.attrib}
            )
    return alarms


def _text(element: ET.Element, tag: str) -> str | None:
    child = element.find(tag)
    if child is None or child.text is None:
        return None
    text = child.text.strip()
    return text or None


def _replace_extra(entry: DictionaryEntry, extra: dict[str, Any]) -> DictionaryEntry:
    return DictionaryEntry(
        series_key=entry.series_key,
        name=entry.name,
        dictionary_type=entry.dictionary_type,
        description=entry.description,
        units_eu=entry.units_eu,
        units_dn=entry.units_dn,
        extra=extra,
    )
