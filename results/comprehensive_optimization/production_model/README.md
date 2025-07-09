# LSTM v2 Production Model

## Model Information
- **Model Type**: LSTM v2 Hybrid (Sensor + Demographics)
- **Window Configuration**: w64_s16
- **Fusion Method**: attention
- **CMI Score**: 0.7929

## Performance Metrics
- Binary F1 Score: 0.9491
- Macro F1 Score: 0.6366
- Test Accuracy: 0.6062

## Model Architecture
- LSTM Units: 80, 64
- Dense Units: 48
- Dropout Rate: 0.15
- Learning Rate: 3.6e-03
- Batch Size: 32

## Files
- `best_lstm_v2_model.h5`: Trained model weights
- `model_config.json`: Complete model configuration
- `comprehensive_optimization_summary.json`: Optimization results

## Usage
```python
from src.lstm_v2_trainer import LSTMv2Trainer

# Load the trained model
trainer = LSTMv2Trainer(
    experiment_name="production",
    window_config="w64_s16",
    n_demographics_features=18
)

# Load model weights
trainer.model.load_model("../results/comprehensive_optimization/production_model/best_lstm_v2_model.h5")
```

Generated on: 2025-07-09 00:42:15
