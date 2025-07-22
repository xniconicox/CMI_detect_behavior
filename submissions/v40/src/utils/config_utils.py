import yaml
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config_v2.yaml"


def load_config(config_path=None):
    """Load configuration from specified path or default path.
    
    Args:
        config_path: Path to config file. If None, uses default config.
        
    Returns:
        dict: Configuration dictionary
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_preprocessing_params(config_path=None):
    """Return preprocessing-related hyperparameters from config.
    
    Args:
        config_path: Path to config file. If None, uses default config.
        
    Returns:
        dict: Preprocessing parameters
    """
    cfg = load_config(config_path)
    return cfg.get("preprocessing", {})


def get_cache_dir(cfg: dict | None = None, config_path=None) -> Path:
    """Return cache directory Path from config.
    
    Args:
        cfg: Configuration dictionary. If None, loads from config_path.
        config_path: Path to config file. If None, uses default config.
        
    Returns:
        Path: Cache directory path
    """
    cfg = cfg or load_config(config_path)
    return Path(cfg.get("cache_dir", "cache"))
