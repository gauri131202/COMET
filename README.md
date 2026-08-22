# COMET

COMET is a Python tool for working through a telemetry anomaly workflow (detection and diagnosis). Specifically, COMET:

1. Ingests CSV telemetry
2. Profiles the dataset
3. Prepares model-ready tables
4. Trains an LSTM autoencoder
5. Detects anomalies
6. Uses mission documents and RAG to help diagnose detected anomalies


## Project Layout

```text
.
├── config/                 Configuration files used by the pipeline
│   ├── model_configs/      Training settings, including the default LSTM config
│   └── prompt_templates/   Prompt text used for RAG diagnosis
├── data/                   Put telemetry CSV files here
├── db/                     Local SQLite database is created here
├── diagnostics/            Model diagnostic output is written here
├── models/                 Trained model files are written here
├── rag/                    Mission document ingestion and vector store output
│   └── documents/          Put mission docs here before running ingest-docs
├── src/                    Application code
│   ├── anomaly/            Detection and anomaly packet creation
│   ├── cleaning/           Dataset preparation and feature config handling
│   ├── cli/                Command-line entry point
│   ├── common/             Shared paths and logging helpers
│   ├── config/             App config loading
│   ├── evaluation/         Evaluation against labeled anomaly intervals
│   ├── ingestion/          CSV ingestion
│   ├── modeling/           LSTM model training and diagnostics
│   ├── profiling/          Dataset profiling and feature config generation
│   ├── rag/                Document ingestion and diagnosis support
│   └── telemetry/          Telemetry record conversion helpers
├── pyproject.toml          Python package metadata
└── requirements.txt        Runtime Python dependencies
```

## Prerequisites

Use Python 3.10 or newer.

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

or on Linux:

```bash
python -m venv venv
venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

For anomaly diagnosis, install Ollama. Downloads can be found here: https://ollama.com/download.

The default diagnosis command uses `gemma:2b` unless you pass another model name.

## Add Telemetry Data

Put one or more CSV files in:

```text
data/
```

The CSV can be CATs data or real mission telemetry. The repository does not include the CATs dataset. If you are using CATs, download it from its original source, keep the telemetry CSV in `data/`, and keep any labeled anomaly metadata available for the optional evaluation step.

The CATs dataset can be found here: https://huggingface.co/datasets/patrickfleith/controlled-anomalies-time-series-dataset.

The current ingestion command reads every `.csv` file directly under `data/`.

## Configure The Model

The default training config is:

```text
config/model_configs/default.yaml
```

You can use it as-is for a first run. Adjust it if you want to change the window size, stride, hidden size, number of epochs, batch size, learning rate, or threshold percentile.

## Add Mission Documents

For diagnosis, add mission documents to:

```text
rag/documents/
```

Supported document types are `.txt`, `.md`, and `.pdf`.

Mission documents are part of the diagnosis workflow. Add or update them before running `ingest-docs`.

## Run The Pipeline

Run commands from the repository root.

### 1. Ingest CSV Files

```bash
python -m src.cli ingest
```

This creates the SQLite database in `db/` if it does not already exist, registers the CSV files, and writes raw dataset tables.

The command prints the number of files ingested. Note the `dataset_id` in later command output or inspect the database if needed.

### 2. Profile The Dataset

```bash
python -m src.cli profile --dataset-id 1
```

This profiles the raw dataset and writes a generated feature config like:

```text
config/generated/feature_config_dataset_1.yaml
```

### 3. Prepare Model-Ready Tables

```bash
python -m src.cli prepare --dataset-id 1
```

By default this uses the generated feature config for that dataset.

To use a specific feature config:

```bash
python -m src.cli prepare --dataset-id 1 --feature-config path/to/feature_config.yaml
```

### 4. Train A Model

```bash
python -m src.cli train --dataset-id 1
```

By default this uses:

```text
config/model_configs/default.yaml
```

To use another model config:

```bash
python -m src.cli train --dataset-id 1 --config path/to/model_config.yaml
```

The command prints a `model_id` and writes the trained model file under `models/`.

### 5. Ingest Mission Documents

```bash
python -m src.cli ingest-docs
```

This reads documents from `rag/documents/` and creates a local vector store under `rag/vector_store/`.

You can adjust chunking:

```bash
python -m src.cli ingest-docs --chunk-size 500 --chunk-overlap 50
```

### 6. Detect Anomalies

```bash
python -m src.cli detect --dataset-id 1 --model-id 1
```

Use the `model_id` printed by the training step.

### 7. List Detected Anomalies

```bash
python -m src.cli list-anomalies --dataset-id 1 --model-id 1
```

You can also limit the number of rows:

```bash
python -m src.cli list-anomalies --dataset-id 1 --model-id 1 --limit 10
```

### 8. Diagnose An Anomaly

```bash
python -m src.cli diagnose --anomaly-id 1
```

To use a different Ollama model:

```bash
python -m src.cli diagnose --anomaly-id 1 --llm-model llama3.2
```

Diagnosis requires both a stored anomaly and ingested mission documents.

### 9. Inspect An Anomaly Packet (Optional)

```bash
python -m src.cli anomaly-packet --anomaly-id 1
```

Use an anomaly ID from `list-anomalies`.

This prints the JSON packet for an existing anomaly. Use it to inspect the anomaly score, time window, contributing features, and surrounding context.

### 10. Run Model Diagnostics (Optional)

```bash
python -m src.cli model-diagnostics --dataset-id 1 --model-id 1
```

This writes diagnostic summaries and plots under `diagnostics/`.

### 11. Check Model Performance (Optional)

If you have a metadata file with known anomaly time intervals, COMET can compare its detections to those known anomalies and report detection metrics.

```bash
python -m src.cli evaluate --dataset-id 1 --model-id 1 --metadata path/to/metadata.csv
```

For CATs, use the metadata file that comes with the dataset download. Keep it outside the repository or in a local ignored folder.

You can choose where evaluation output goes:

```bash
python -m src.cli evaluate --dataset-id 1 --model-id 1 --metadata path/to/metadata.csv --output-dir evaluation/run_1
```
