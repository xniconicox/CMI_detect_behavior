#!/bin/bash

# Training script for v52 model
set -e

EXPERIMENT_NAME="preprocess_v52"
EPOCHS=10
BATCH_SIZE=32

cd "$(dirname "$0")/.."

python - <<'PY'
from src.trainers.multimodal_trainer_v52 import MultimodalTrainerV52
trainer = MultimodalTrainerV52("preprocess_v52")
data = trainer.load_all_data()
res = trainer.train(data, epochs=$EPOCHS, batch_size=$BATCH_SIZE)
print(res)
trainer.save_model()
trainer.save_history()
PY
