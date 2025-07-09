import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def get_preprocessing_params():
    """Return preprocessing-related hyperparameters from config."""
    cfg = load_config()
    return cfg.get("preprocessing", {})
