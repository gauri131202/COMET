from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.anomaly.packet import build_anomaly_packet
from src.common.paths import CONFIG_DIR, RAG_VECTOR_DIR
from src.db import init_db, insert_diagnosis
from src.rag.document_ingestion import _VectorStore


DEFAULT_PROMPT_TEMPLATE = CONFIG_DIR / "prompt_templates" / "default.txt"
DEFAULT_OLLAMA_MODEL = "gemma:2b"
DEFAULT_TOP_K = 5


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: str
    path: str
    text: str


def diagnose_anomaly(
    anomaly_id: str,
    *,
    prompt_template_path: str | Path = DEFAULT_PROMPT_TEMPLATE,
    vector_store_dir: str | Path = RAG_VECTOR_DIR,
    llm_model: str = DEFAULT_OLLAMA_MODEL,
    top_k: int = DEFAULT_TOP_K,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    init_db(db_path)
    packet = build_anomaly_packet(anomaly_id, db_path=db_path)
    retrieved_chunks = retrieve_relevant_chunks(packet, top_k=top_k, vector_store_dir=vector_store_dir)
    prompt_template = load_prompt_template(prompt_template_path)
    prompt = build_prompt(packet, retrieved_chunks, prompt_template)
    response_text, llm_status = call_ollama(prompt, model=llm_model)
    if response_text is None:
        response_text = _fallback_response(packet, retrieved_chunks, llm_status)

    diagnosis_id = insert_diagnosis(
        anomaly_id=anomaly_id,
        prompt_template_name=prompt_template.name,
        prompt_template_version=prompt_template.version,
        llm_model=llm_model,
        retrieved_chunks=[
            {
                "chunk_id": chunk["chunk_id"],
                "source_file": chunk["source_file"],
                "score": chunk["score"],
            }
            for chunk in retrieved_chunks
        ],
        response_text=response_text,
        metadata={
            "prompt_template_path": prompt_template.path,
            "retrieval_top_k": top_k,
            "llm_status": llm_status,
            "chunk_count": len(retrieved_chunks),
        },
        db_path=db_path,
    )

    return {
        "diagnosis_id": diagnosis_id,
        "anomaly_id": anomaly_id,
        "prompt_template": {
            "name": prompt_template.name,
            "version": prompt_template.version,
            "path": prompt_template.path,
        },
        "llm_model": llm_model,
        "llm_status": llm_status,
        "retrieved_chunks": retrieved_chunks,
        "response_text": response_text,
    }


def retrieve_relevant_chunks(
    anomaly_packet: dict[str, Any],
    *,
    top_k: int = DEFAULT_TOP_K,
    vector_store_dir: str | Path = RAG_VECTOR_DIR,
) -> list[dict[str, Any]]:
    query_text = _packet_query_text(anomaly_packet)
    return _VectorStore(Path(vector_store_dir)).query(query_text, top_k)


def load_prompt_template(path: str | Path = DEFAULT_PROMPT_TEMPLATE) -> PromptTemplate:
    template_path = Path(path)
    if not template_path.exists():
        template_path.parent.mkdir(parents=True, exist_ok=True)
        template_path.write_text(_default_prompt_template_text(), encoding="utf-8")
    text = template_path.read_text(encoding="utf-8")
    return PromptTemplate(
        name=template_path.stem,
        version="1",
        path=str(template_path),
        text=text,
    )


def build_prompt(
    anomaly_packet: dict[str, Any],
    retrieved_chunks: list[dict[str, Any]],
    prompt_template: PromptTemplate,
) -> str:
    compact_packet = _compact_packet(anomaly_packet)
    compact_chunks = [
        {
            "chunk_id": chunk["chunk_id"],
            "source_file": chunk["source_file"],
            "document_version": chunk["document_version"],
            "score": round(float(chunk["score"]), 6),
            "text": chunk["text"],
        }
        for chunk in retrieved_chunks
    ]
    telemetry_tools = [
        "get_raw_window(dataset_id, start_row_id, end_row_id)",
        "get_context_window(dataset_id, start_row_id, end_row_id)",
        "get_column_summary(dataset_id, column_name, start_row_id, end_row_id)",
        "get_rows_around_peak(dataset_id, anomaly_id)",
    ]
    return prompt_template.text.format(
        anomaly_packet=json.dumps(compact_packet, indent=2, sort_keys=True),
        retrieved_chunks=json.dumps(compact_chunks, indent=2, sort_keys=True),
        telemetry_tools="\n".join(f"- {tool}" for tool in telemetry_tools),
    )


def call_ollama(
    prompt: str,
    *,
    model: str = DEFAULT_OLLAMA_MODEL,
    timeout_seconds: int = 30,
) -> tuple[str | None, str]:
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            parsed = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return None, f"ollama_unavailable: {exc}"
    generated = parsed.get("response")
    if not generated:
        return None, "ollama_empty_response"
    return str(generated), "ollama_success"


def _fallback_response(
    anomaly_packet: dict[str, Any],
    retrieved_chunks: list[dict[str, Any]],
    llm_status: str,
) -> str:
    if not retrieved_chunks:
        evidence_note = "No mission document chunks were retrieved."
    else:
        refs = ", ".join(chunk["chunk_id"] for chunk in retrieved_chunks)
        evidence_note = f"Retrieved mission document chunks: {refs}."

    top_features = [
        item.get("feature")
        for item in anomaly_packet.get("top_contributing_features", [])
        if isinstance(item, dict)
    ]
    return (
        "Local LLM diagnosis was not generated because Ollama was unavailable "
        f"({llm_status}). The anomaly packet indicates score "
        f"{anomaly_packet.get('anomaly_score')} against threshold {anomaly_packet.get('threshold')}. "
        f"Top contributing features: {top_features}. {evidence_note} "
        "Additional information needed: run Ollama locally with the configured model or review the "
        "retrieved mission chunks and controlled telemetry summaries manually."
    )


def _packet_query_text(packet: dict[str, Any]) -> str:
    parts = [
        f"anomaly {packet.get('anomaly_id')}",
        f"score {packet.get('anomaly_score')} threshold {packet.get('threshold')}",
        json.dumps(packet.get("top_contributing_features", []), sort_keys=True),
        json.dumps(packet.get("numeric_summary", {}), sort_keys=True),
        json.dumps(packet.get("categorical_changes", {}), sort_keys=True),
    ]
    return "\n".join(parts)


def _compact_packet(packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "anomaly_id": packet.get("anomaly_id"),
        "dataset_id": packet.get("dataset_id"),
        "model_id": packet.get("model_id"),
        "time_window": packet.get("time_window"),
        "anomaly_score": packet.get("anomaly_score"),
        "threshold": packet.get("threshold"),
        "model_context": packet.get("model_context"),
        "numeric_summary": packet.get("numeric_summary"),
        "categorical_changes": packet.get("categorical_changes"),
        "top_contributing_features": packet.get("top_contributing_features"),
        "raw_data_reference": packet.get("raw_data_reference"),
    }


def _default_prompt_template_text() -> str:
    return """You are a satellite operations assistant helping diagnose telemetry anomalies.

Use only the provided anomaly packet, telemetry summaries, and retrieved mission documentation.
If the evidence is insufficient, say what additional information is needed.

Anomaly Packet:
{anomaly_packet}

Retrieved Mission Documentation:
{retrieved_chunks}

Controlled Telemetry Tools Available:
{telemetry_tools}

Return:
1. Summary of the anomaly
2. Most likely subsystem or behavior involved
3. Evidence from telemetry
4. Relevant documentation references
5. Recommended next actions
6. Additional data needed, if any
"""
