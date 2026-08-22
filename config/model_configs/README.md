# config/model_configs

Model training configuration files.

Overview:
- `default.yaml`: change the LSTM window size, stride, hidden size, number of layers, epochs, batch size, learning rate, or anomaly threshold percentile.

If needed, you can use another file by passing `--config path/to/config.yaml` to `python -m src.cli train`.