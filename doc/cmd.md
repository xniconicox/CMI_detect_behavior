

python -m scripts.run_preprocessing   --experiment-name 20250717_preproc_train   --config config/config_v2.yaml   --use-cache   --mode train

bash scripts/run_multimodal_training_v20.sh

python -m src.scripts.visualize_training_history --history output/experiments/20250717_preproc_train/results/training_history.json   --save output/experiments/20250717_preproc_pipeline_train/results/training_history_XXXX.png   --title "20250717_preproc_pipeline_first_try"




git pull origin develop_local

git push origin develop_local --force-with-lease

git fetch --prune