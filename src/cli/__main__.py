from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from typing import Any

from src.anomaly.detect import detect_anomalies
from src.anomaly.packet import build_anomaly_packet
from src.cleaning.prepare_dataset import prepare_dataset, resolve_feature_config_path
from src.cli.prerequisites import (
    require_diagnosis_inputs,
    require_documents_directory,
    require_model_ready_dataset,
    require_profile_output,
    require_raw_dataset,
    require_trained_model,
)
from src.common.logging import get_app_logger
from src.db import get_connection
from src.evaluation.anomaly_evaluation import evaluate_anomalies
from src.ingestion.csv_ingestion import ingest_csvs
from src.modeling.diagnostics import run_model_diagnostics
from src.modeling.train_model import resolve_training_config_path, train_model
from src.profiling.dataset_profile import profile_dataset
from src.rag.diagnosis import diagnose_anomaly
from src.rag.document_ingestion import ingest_documents


def _run_stage(
    parser: argparse.ArgumentParser,
    stage: str,
    action: Callable[[], Any],
) -> Any:
    logger = get_app_logger()
    logger.info("stage=%s status=started", stage)
    try:
        result = action()
    except ValueError as exc:
        logger.error("stage=%s status=failed error=%s", stage, exc)
        parser.exit(1, f"error: {exc}\n")
    except Exception:
        logger.exception("stage=%s status=failed", stage)
        raise
    logger.info("stage=%s status=succeeded", stage)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m src.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("ingest", help="Ingest raw CSV files from data/.")
    profile_parser = subparsers.add_parser("profile", help="Profile an ingested dataset.")
    profile_parser.add_argument("--dataset-id", type=int, required=True)
    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Create model/context-ready tables using the generated feature config by default.",
    )
    prepare_parser.add_argument("--dataset-id", type=int, required=True)
    prepare_parser.add_argument(
        "--feature-config",
        help=(
            "Optional path to a feature config. Defaults to "
            "config/generated/feature_config_dataset_<dataset_id>.yaml."
        ),
    )
    train_parser = subparsers.add_parser("train", help="Train an LSTM autoencoder.")
    train_parser.add_argument("--dataset-id", type=int, required=True)
    train_parser.add_argument(
        "--config",
        help="Optional training config path. Defaults to config/model_configs/default.yaml.",
    )
    detect_parser = subparsers.add_parser("detect", help="Detect anomalies with a trained model.")
    detect_parser.add_argument("--dataset-id", type=int, required=True)
    detect_parser.add_argument("--model-id", type=int, required=True)
    list_anomalies_parser = subparsers.add_parser("list-anomalies", help="List detected anomaly IDs.")
    list_anomalies_parser.add_argument("--dataset-id", type=int)
    list_anomalies_parser.add_argument("--model-id", type=int)
    list_anomalies_parser.add_argument("--limit", type=int, default=20)
    diagnostics_parser = subparsers.add_parser(
        "model-diagnostics",
        help="Inspect reconstruction errors, threshold behavior, and feature contributions for a trained model.",
    )
    diagnostics_parser.add_argument("--dataset-id", type=int, required=True)
    diagnostics_parser.add_argument("--model-id", type=int, required=True)
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Compare detected anomalies against labeled anomaly intervals and write evaluation plots.",
    )
    evaluate_parser.add_argument("--dataset-id", type=int, required=True)
    evaluate_parser.add_argument("--model-id", type=int, required=True)
    evaluate_parser.add_argument("--metadata", required=True, help="Path to metadata CSV with labeled anomaly intervals.")
    evaluate_parser.add_argument("--output-dir", help="Optional output directory for summary JSON and plots.")
    packet_parser = subparsers.add_parser("anomaly-packet", help="Print an anomaly packet as JSON.")
    packet_parser.add_argument("--anomaly-id", required=True)
    docs_parser = subparsers.add_parser("ingest-docs", help="Ingest local mission documents.")
    docs_parser.add_argument("--chunk-size", type=int, default=500)
    docs_parser.add_argument("--chunk-overlap", type=int, default=50)
    diagnose_parser = subparsers.add_parser("diagnose", help="Run RAG diagnosis for an anomaly.")
    diagnose_parser.add_argument("--anomaly-id", required=True)
    diagnose_parser.add_argument(
        "--llm-model",
        default="gemma:2b",
        help="Ollama model name to use for diagnosis. Defaults to gemma:2b.",
    )

    args = parser.parse_args()

    if args.command == "ingest":
        summary = _run_stage(parser, "ingest", ingest_csvs)
        print(
            "CSV ingestion complete: "
            f"{summary['ingested']} ingested, "
            f"{summary['skipped_duplicates']} duplicates skipped, "
            f"{summary['failed']} failed."
        )
    elif args.command == "profile":
        def action() -> dict[str, Any]:
            require_raw_dataset(args.dataset_id)
            return profile_dataset(args.dataset_id)

        summary = _run_stage(parser, "profile", action)
        print(
            "Dataset profiling complete: "
            f"dataset_id={summary['dataset_id']} "
            f"columns={summary['column_count']} "
            f"config={summary['config_path']}"
        )
    elif args.command == "prepare":
        def action() -> dict[str, Any]:
            require_profile_output(args.dataset_id, args.feature_config)
            return prepare_dataset(args.dataset_id, args.feature_config)

        summary = _run_stage(parser, "prepare", action)
        config_path = resolve_feature_config_path(args.dataset_id, args.feature_config)
        print(
            "Dataset preparation complete: "
            f"dataset_id={summary['dataset_id']} "
            f"model_table={summary['model_table']} "
            f"context_table={summary['context_table']} "
            f"rows={summary['row_count']} "
            f"config={config_path}"
        )
    elif args.command == "train":
        def action() -> dict[str, Any]:
            require_model_ready_dataset(args.dataset_id)
            return train_model(args.dataset_id, args.config)

        summary = _run_stage(parser, "train", action)
        config_path = resolve_training_config_path(args.config)
        print(
            "Model training complete: "
            f"model_id={summary['model_id']} "
            f"dataset_id={summary['dataset_id']} "
            f"threshold={summary['threshold']:.6f} "
            f"path={summary['model_path']} "
            f"config={config_path}"
        )
    elif args.command == "detect":
        def action() -> dict[str, Any]:
            require_trained_model(args.dataset_id, args.model_id)
            return detect_anomalies(args.dataset_id, args.model_id)

        summary = _run_stage(parser, "detect", action)
        print(
            "Anomaly detection complete: "
            f"dataset_id={summary['dataset_id']} "
            f"model_id={summary['model_id']} "
            f"windows={summary['windows_scored']} "
            f"anomalies={summary['anomalies_detected']}"
        )
    elif args.command == "list-anomalies":
        rows = _run_stage(
            parser,
            "list-anomalies",
            lambda: list_anomalies(
                dataset_id=args.dataset_id,
                model_id=args.model_id,
                limit=args.limit,
            ),
        )
        if not rows:
            print("No anomalies found. Run detect first.")
        else:
            print("anomaly_id\tdataset_id\tmodel_id\twindow_index\tscore\tstart_timestamp\tend_timestamp")
            for row in rows:
                print(
                    f"{row['anomaly_id']}\t"
                    f"{row['dataset_id']}\t"
                    f"{row['model_id']}\t"
                    f"{row['window_index']}\t"
                    f"{row['anomaly_score']:.6g}\t"
                    f"{row['start_timestamp'] or ''}\t"
                    f"{row['end_timestamp'] or ''}"
                )
    elif args.command == "model-diagnostics":
        def action() -> dict[str, Any]:
            require_trained_model(args.dataset_id, args.model_id)
            return run_model_diagnostics(args.dataset_id, args.model_id)

        summary = _run_stage(parser, "model-diagnostics", action)
        stats = summary["reconstruction_error_statistics"]
        top_features = ", ".join(
            f"{item['feature']}={item['average_reconstruction_error']:.6g}"
            for item in summary["top_contributing_features"][:5]
        ) or "none"
        print(
            "Model diagnostics complete: "
            f"dataset_id={summary['dataset_id']} "
            f"model_id={summary['model_id']} "
            f"windows={summary['windows_evaluated']} "
            f"above_threshold={summary['windows_above_threshold']} "
            f"percent_above={summary['percentage_windows_above_threshold']:.2f}% "
            f"threshold={summary['threshold']:.6g} "
            f"threshold_percentile={summary['threshold_percentile']} "
            f"min={stats['min']:.6g} "
            f"max={stats['max']:.6g} "
            f"mean={stats['mean']:.6g} "
            f"median={stats['median']:.6g} "
            f"std={stats['standard_deviation']:.6g} "
            f"top_features={top_features} "
            f"summary={summary['summary_path']}"
        )
    elif args.command == "evaluate":
        def action() -> dict[str, Any]:
            require_trained_model(args.dataset_id, args.model_id)
            if args.output_dir:
                return evaluate_anomalies(
                    args.dataset_id,
                    args.model_id,
                    args.metadata,
                    output_dir=args.output_dir,
                )
            return evaluate_anomalies(args.dataset_id, args.model_id, args.metadata)

        summary = _run_stage(parser, "evaluate", action)
        print(
            "Evaluation complete: "
            f"dataset_id={summary['dataset_id']} "
            f"model_id={summary['model_id']} "
            f"ground_truth={summary['ground_truth_intervals']} "
            f"detections={summary['detected_anomalies']} "
            f"precision={summary['precision']:.4f} "
            f"recall={summary['recall']:.4f} "
            f"f1={summary['f1']:.4f} "
            f"mean_delay={summary['mean_detection_delay_seconds']} "
            f"median_delay={summary['median_detection_delay_seconds']} "
            f"summary={summary['summary_path']}"
        )
    elif args.command == "anomaly-packet":
        packet = _run_stage(
            parser,
            "anomaly-packet",
            lambda: build_anomaly_packet(args.anomaly_id),
        )
        print(json.dumps(packet, indent=2, sort_keys=True))
    elif args.command == "ingest-docs":
        def action() -> dict[str, Any]:
            require_documents_directory()
            return ingest_documents(
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )

        summary = _run_stage(parser, "ingest-docs", action)
        print(
            "Document ingestion complete: "
            f"found={summary['documents_found']} "
            f"ingested={summary['documents_ingested']} "
            f"skipped={summary['documents_skipped']} "
            f"chunks={summary['chunks_ingested']} "
            f"vector_store={summary['vector_store']}"
        )
    elif args.command == "diagnose":
        def action() -> dict[str, Any]:
            require_diagnosis_inputs(args.anomaly_id)
            return diagnose_anomaly(args.anomaly_id, llm_model=args.llm_model)

        summary = _run_stage(parser, "diagnose", action)
        print(
            "Diagnosis complete: "
            f"diagnosis_id={summary['diagnosis_id']} "
            f"anomaly_id={summary['anomaly_id']} "
            f"chunks={len(summary['retrieved_chunks'])} "
            f"llm_status={summary['llm_status']}"
        )


def list_anomalies(
    *,
    dataset_id: int | None = None,
    model_id: int | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    if limit <= 0:
        raise ValueError("limit must be positive.")
    clauses = []
    params: list[Any] = []
    if dataset_id is not None:
        clauses.append("dataset_id = ?")
        params.append(dataset_id)
    if model_id is not None:
        clauses.append("model_id = ?")
        params.append(model_id)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(limit)
    with get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT anomaly_id, dataset_id, model_id, window_index, anomaly_score,
                   start_timestamp, end_timestamp
            FROM anomalies
            {where}
            ORDER BY anomaly_score DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
    return [dict(row) for row in rows]


if __name__ == "__main__":
    main()
