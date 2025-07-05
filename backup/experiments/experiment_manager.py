import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd

class ExperimentManager:
    """実験管理クラス"""
    def __init__(self, output_dir: str = "experiments"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.current_experiment_dir: Optional[Path] = None
        self.current_experiment_name: Optional[str] = None

    def create_experiment(self, experiment_name: str, config_file: Optional[str] = None) -> Dict[str, Any]:
        """新しい実験を作成"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = f"exp_{timestamp}_{experiment_name}"
        exp_dir = self.output_dir / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        if config_file:
            config_path = Path(config_file)
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = self._get_default_config()
        experiment_info = {
            "experiment_id": exp_id,
            "experiment_name": experiment_name,
            "created_at": datetime.now().isoformat(),
            "status": "created",
            "config": config,
            "results": {}
        }
        with open(exp_dir / "experiment_info.json", 'w') as f:
            json.dump(experiment_info, f, indent=2)
        print(f"実験作成完了: {exp_id}")
        return experiment_info

    def load_experiment(self, experiment_id: str) -> Dict[str, Any]:
        exp_dir = self.output_dir / experiment_id
        if not exp_dir.exists():
            raise FileNotFoundError(f"実験が見つかりません: {experiment_id}")
        with open(exp_dir / "experiment_info.json", 'r') as f:
            return json.load(f)

    def update_experiment(self, experiment_id: str, results: Dict[str, Any]):
        exp_dir = self.output_dir / experiment_id
        info_file = exp_dir / "experiment_info.json"
        with open(info_file, 'r') as f:
            experiment_info = json.load(f)
        experiment_info["results"].update(results)
        experiment_info["updated_at"] = datetime.now().isoformat()
        with open(info_file, 'w') as f:
            json.dump(experiment_info, f, indent=2)

    def list_experiments(self) -> List[Dict[str, Any]]:
        experiments = []
        for exp_dir in self.output_dir.iterdir():
            if exp_dir.is_dir():
                info_file = exp_dir / "experiment_info.json"
                if info_file.exists():
                    with open(info_file, 'r') as f:
                        exp_info = json.load(f)
                    experiments.append({
                        "id": exp_info["experiment_id"],
                        "name": exp_info["experiment_name"],
                        "created_at": exp_info["created_at"],
                        "status": exp_info["status"],
                        "cv_score": exp_info.get("results", {}).get("cv_score", "N/A")
                    })
        return sorted(experiments, key=lambda x: x["created_at"], reverse=True)

    def start_experiment(self, experiment_name: str) -> None:
        """実験を開始"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = f"{experiment_name}_{timestamp}"
        self.current_experiment_dir = self.output_dir / exp_id
        self.current_experiment_name = experiment_name
        
        # 実験ディレクトリを作成
        self.current_experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 実験情報を記録
        experiment_info = {
            "experiment_name": experiment_name,
            "started_at": datetime.now().isoformat(),
            "status": "running",
            "metrics": {},
            "parameters": {}
        }
        
        with open(self.current_experiment_dir / "experiment_info.json", 'w') as f:
            json.dump(experiment_info, f, indent=2)
        
        print(f"実験開始: {exp_id}")

    def get_experiment_dir(self) -> Path:
        """現在の実験ディレクトリを取得"""
        if self.current_experiment_dir is None:
            raise ValueError("実験が開始されていません")
        return self.current_experiment_dir

    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """メトリクスを記録"""
        if self.current_experiment_dir is None:
            raise ValueError("実験が開始されていません")
        
        info_file = self.current_experiment_dir / "experiment_info.json"
        with open(info_file, 'r') as f:
            experiment_info = json.load(f)
        
        experiment_info["metrics"].update(metrics)
        experiment_info["updated_at"] = datetime.now().isoformat()
        
        with open(info_file, 'w') as f:
            json.dump(experiment_info, f, indent=2)

    def log_parameters(self, parameters: Dict[str, Any]) -> None:
        """パラメータを記録"""
        if self.current_experiment_dir is None:
            raise ValueError("実験が開始されていません")
        
        info_file = self.current_experiment_dir / "experiment_info.json"
        with open(info_file, 'r') as f:
            experiment_info = json.load(f)
        
        experiment_info["parameters"].update(parameters)
        experiment_info["updated_at"] = datetime.now().isoformat()
        
        with open(info_file, 'w') as f:
            json.dump(experiment_info, f, indent=2)

    def log_error(self, error_message: str) -> None:
        """エラーを記録"""
        if self.current_experiment_dir is None:
            raise ValueError("実験が開始されていません")
        
        info_file = self.current_experiment_dir / "experiment_info.json"
        with open(info_file, 'r') as f:
            experiment_info = json.load(f)
        
        experiment_info["status"] = "error"
        experiment_info["error"] = error_message
        experiment_info["updated_at"] = datetime.now().isoformat()
        
        with open(info_file, 'w') as f:
            json.dump(experiment_info, f, indent=2)

    def end_experiment(self) -> None:
        """実験を終了"""
        if self.current_experiment_dir is None:
            raise ValueError("実験が開始されていません")
        
        info_file = self.current_experiment_dir / "experiment_info.json"
        with open(info_file, 'r') as f:
            experiment_info = json.load(f)
        
        experiment_info["status"] = "completed"
        experiment_info["completed_at"] = datetime.now().isoformat()
        
        with open(info_file, 'w') as f:
            json.dump(experiment_info, f, indent=2)
        
        print(f"実験終了: {self.current_experiment_name}")

    def compare_experiments(self, experiment_ids: List[str]) -> pd.DataFrame:
        comparison_data = []
        for exp_id in experiment_ids:
            exp_info = self.load_experiment(exp_id)
            results = exp_info.get("results", {})
            comparison_data.append({
                "experiment_id": exp_id,
                "experiment_name": exp_info["experiment_name"],
                "cv_score": results.get("cv_score", "N/A"),
                "cv_std": results.get("cv_std", "N/A"),
                "best_params": results.get("best_params", "N/A"),
                "feature_count": results.get("feature_count", "N/A")
            })
        return pd.DataFrame(comparison_data)

    def _get_default_config(self) -> Dict[str, Any]:
        return {
            "model": {
                "type": "lightgbm",
                "params": {
                    "objective": "multiclass",
                    "metric": "multi_logloss",
                    "boosting_type": "gbdt",
                    "num_leaves": 31,
                    "learning_rate": 0.05,
                    "feature_fraction": 0.9,
                    "bagging_fraction": 0.8,
                    "bagging_freq": 5,
                    "verbose": -1,
                    "random_state": 42
                }
            },
            "training": {
                "n_folds": 5,
                "random_state": 42
            },
            "preprocessing": {
                "feature_selection": False,
                "feature_threshold": 100
            }
        }
