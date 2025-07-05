"""
実験設定管理クラス
"""
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """モデル設定"""
    model_type: str = "lightgbm"
    objective: str = "multiclass"
    num_class: int = 3
    random_state: int = 42
    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: int = 6
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0
    class_weight: Optional[str] = None
    device: str = "cpu"  # "cpu" or "gpu"


@dataclass
class DataConfig:
    """データ設定"""
    data_dir: str = "data"
    train_file: str = "train.csv"
    test_file: str = "test.csv"
    target_column: str = "target"
    feature_columns: Optional[list] = None
    test_size: float = 0.2
    random_state: int = 42
    preprocessing: Optional[Dict[str, Any]] = None


@dataclass
class TrainingConfig:
    """学習設定"""
    cv_folds: int = 5
    cv_strategy: str = "stratified"  # "stratified", "kfold", "group"
    group_column: Optional[str] = None
    scoring: str = "f1_macro"
    n_jobs: int = -1
    verbose: int = 1
    early_stopping_rounds: int = 50
    eval_metric: str = "multi_logloss"


@dataclass
class ExperimentConfig:
    """実験設定"""
    experiment_name: str
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    output_dir: str = "experiments"
    save_model: bool = True
    save_predictions: bool = True
    save_feature_importance: bool = True
    save_confusion_matrix: bool = True


class ConfigManager:
    """設定管理クラス"""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_path = Path(config_path) if config_path else None
        self.config: Optional[ExperimentConfig] = None
    
    def create_default_config(self, experiment_name: str) -> ExperimentConfig:
        """デフォルト設定を作成"""
        return ExperimentConfig(
            experiment_name=experiment_name,
            model=ModelConfig(),
            data=DataConfig(),
            training=TrainingConfig()
        )
    
    def load_config(self, config_path: Union[str, Path]) -> ExperimentConfig:
        """設定ファイルを読み込み"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"サポートされていないファイル形式: {config_path.suffix}")
        
        # ネストした辞書をdataclassに変換
        self.config = self._dict_to_config(config_dict)
        self.config_path = config_path
        
        logger.info(f"設定を読み込みました: {config_path}")
        return self.config
    
    def save_config(self, config: ExperimentConfig, output_path: Union[str, Path]) -> None:
        """設定をファイルに保存"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = asdict(config)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            if output_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, indent=2)
            elif output_path.suffix.lower() == '.json':
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"サポートされていないファイル形式: {output_path.suffix}")
        
        logger.info(f"設定を保存しました: {output_path}")
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> ExperimentConfig:
        """辞書をExperimentConfigに変換"""
        # ネストした設定を適切なdataclassに変換
        model_config = ModelConfig(**config_dict.get('model', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        
        return ExperimentConfig(
            experiment_name=config_dict['experiment_name'],
            model=model_config,
            data=data_config,
            training=training_config,
            output_dir=config_dict.get('output_dir', 'experiments'),
            save_model=config_dict.get('save_model', True),
            save_predictions=config_dict.get('save_predictions', True),
            save_feature_importance=config_dict.get('save_feature_importance', True),
            save_confusion_matrix=config_dict.get('save_confusion_matrix', True)
        )
    
    def update_config(self, **kwargs) -> None:
        """設定を更新"""
        if self.config is None:
            raise ValueError("設定が読み込まれていません")
        
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                logger.warning(f"不明な設定キー: {key}")
    
    def get_model_params(self) -> Dict[str, Any]:
        """モデルパラメータを取得"""
        if self.config is None:
            raise ValueError("設定が読み込まれていません")
        
        return asdict(self.config.model)
    
    def get_data_params(self) -> Dict[str, Any]:
        """データパラメータを取得"""
        if self.config is None:
            raise ValueError("設定が読み込まれていません")
        
        return asdict(self.config.data)
    
    def get_training_params(self) -> Dict[str, Any]:
        """学習パラメータを取得"""
        if self.config is None:
            raise ValueError("設定が読み込まれていません")
        
        return asdict(self.config.training) 