# CMI LSTM Models v1

This dataset contains pre-trained LSTM models for the CMI Competition: "Detecting Behavior with Sensor Data".

## Files

- `lstm_best.h5`: Trained LSTM model (TensorFlow/Keras format)
- `lstm_best_architecture.json`: Model architecture definition
- `scaler.pkl`: StandardScaler for feature normalization
- `label_encoder.pkl`: LabelEncoder for gesture classes
- `config.json`: Model configuration parameters
- `meta.json`: Training metadata

## Model Performance

- **F1-macro Score**: 0.5200
- **Accuracy**: 0.5558
- **Window Configuration**: w64_s16 (64 timesteps, stride 16)
- **Architecture**: Bidirectional LSTM
- **Training Data**: 13,393 windows from sensor sequences

## Usage

```python
import tensorflow as tf
import pickle
import json

# Load model
model = tf.keras.models.load_model('lstm_best.h5')

# Load preprocessors
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Load config
with open('config.json', 'r') as f:
    config = json.load(f)
```

## License

All models and code are provided under permissive open-source licenses:
- TensorFlow/Keras: Apache 2.0 License
- Scikit-learn: BSD License
- NumPy, Pandas: BSD License

## Competition

CMI Competition: "Detecting Behavior with Sensor Data"
- 18-class gesture classification
- Sensor data: accelerometer, gyroscope, ToF, thermal
- Time series analysis with sliding windows

## Kaggle Usage

To use this dataset and model on Kaggle, follow these steps:

1. **Upload Dataset to Kaggle**:
   - Go to Kaggle's "Datasets" page and create a new dataset.
   - Upload the following files:
     - `lstm_best.h5`
     - `lstm_best_architecture.json`
     - `scaler.pkl`
     - `label_encoder.pkl`
     - `config.json`
     - `meta.json`

2. **Add Dataset to Kaggle Notebook**:
   - Open your Kaggle notebook.
   - Go to the "Data" tab and add the dataset you uploaded.

3. **Run Inference Script**:
   - Use the following code snippet to load the model and preprocessors in your Kaggle notebook:

   ```python
   import tensorflow as tf
   import pickle
   import json

   # Load model
   model = tf.keras.models.load_model('/kaggle/input/your-dataset-name/lstm_best.h5')

   # Load preprocessors
   with open('/kaggle/input/your-dataset-name/scaler.pkl', 'rb') as f:
       scaler = pickle.load(f)

   with open('/kaggle/input/your-dataset-name/label_encoder.pkl', 'rb') as f:
       label_encoder = pickle.load(f)

   # Load config
   with open('/kaggle/input/your-dataset-name/config.json', 'r') as f:
       config = json.load(f)
   ```

4. **Install Dependencies**:
   - Ensure that all necessary Python packages are installed. You can use the following command in a Kaggle notebook cell:

   ```python
   !pip install tensorflow scikit-learn
   ```

These steps will help you set up and run the model on Kaggle effectively.
