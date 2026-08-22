# src/cleaning

Dataset preparation code.

Overview:
- `prepare_dataset.py`: uses the feature config produced during profiling to build model-ready and context-ready tables

Change feature roles in the generated YAML under `config/generated/` or pass a custom file with `--feature-config`.