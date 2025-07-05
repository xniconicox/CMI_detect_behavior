"""
ベースラインモデル学習スクリプト
"""
import sys
from pathlib import Path
import logging
import argparse
import yaml
import json
import pandas as pd

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from .experiments.experiment_manager import ExperimentManager
from .experiments.config_manager import ConfigManager
from .models.baseline_model import BaselineModel
from .data.data_loader import DataLoader

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='ベースラインモデル学習')
    parser.add_argument('--config', type=str, required=True, help='設定ファイルのパス')
    parser.add_argument('--experiment_name', type=str, help='実験名（設定ファイルで指定されていない場合）')
    parser.add_argument('--data_path', type=str, help='データファイルのパス（設定ファイルで指定されていない場合）')
    parser.add_argument('--output_dir', type=str, help='出力ディレクトリ（設定ファイルで指定されていない場合）')
    
    args = parser.parse_args()
    
    # 設定を読み込み
    config_manager = ConfigManager()
    config = config_manager.load_config(args.config)
    
    # コマンドライン引数で上書き
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    if args.data_path:
        config.data.data_dir = str(Path(args.data_path).parent)
        config.data.train_file = Path(args.data_path).name
    if args.output_dir:
        config.output_dir = args.output_dir
    
    # 実験管理を開始
    exp_manager = ExperimentManager(config.output_dir)
    exp_manager.start_experiment(config.experiment_name)
    
    try:
        # データ読み込み・前処理
        logger.info("データ読み込み・前処理を開始します")
        data_loader = DataLoader(config_manager.get_data_params())
        
        # データパスを構築
        data_path = Path(config.data.data_dir) / config.data.train_file
        
        # データを読み込み
        df = data_loader.load_data(data_path)
        
        # 前処理を実行
        df_processed = data_loader.preprocess_data(df, is_training=True)
        
        # データ分割
        X_train, y_train, X_val, y_val = data_loader.split_data(
            df_processed, target_col=config.data.target_column
        )
        
        # ターゲット変数をエンコード
        y_train_encoded = data_loader.encode_target(y_train, is_training=True)
        y_val_encoded = data_loader.encode_target(y_val, is_training=False)
        
        # モデル学習
        logger.info("モデル学習を開始します")
        model = BaselineModel(config_manager.get_model_params())
        
        # 学習実行
        train_results = model.train(
            X_train, y_train_encoded,
            cv_folds=config.training.cv_folds,
            cv_strategy=config.training.cv_strategy
        )
        
        # 検証データで評価
        eval_results = model.evaluate(X_val, y_val_encoded)
        
        # 結果を記録
        results = {
            'train_results': train_results,
            'eval_results': eval_results,
            'config': config_manager.get_model_params()
        }
        
        exp_manager.log_metrics({
            'cv_mean_f1': train_results['cv_mean'],
            'cv_std_f1': train_results['cv_std'],
            'val_f1_macro': eval_results['f1_macro'],
            'val_f1_weighted': eval_results['f1_weighted']
        })
        
        exp_manager.log_parameters(config_manager.get_model_params())
        
        # ファイル保存
        if config.save_model:
            model_path = exp_manager.get_experiment_dir() / 'model.pkl'
            model.save_model(model_path)
        
        if config.save_predictions:
            # 予測結果を保存
            predictions_df = pd.DataFrame({
                'true': y_val_encoded,
                'predicted': eval_results['predictions'],
                'prob_0': eval_results['probabilities'][:, 0],
                'prob_1': eval_results['probabilities'][:, 1],
                'prob_2': eval_results['probabilities'][:, 2]
            })
            predictions_path = exp_manager.get_experiment_dir() / 'predictions.csv'
            predictions_df.to_csv(predictions_path, index=False)
        
        if config.save_feature_importance:
            # 特徴量重要度を保存
            feature_importance_path = exp_manager.get_experiment_dir() / 'feature_importance.csv'
            train_results['feature_importance'].to_csv(feature_importance_path, index=False)
            
            # 可視化も保存
            plot_path = exp_manager.get_experiment_dir() / 'feature_importance.png'
            model.plot_feature_importance(save_path=plot_path)
        
        if config.save_confusion_matrix:
            # 混同行列を保存
            cm_path = exp_manager.get_experiment_dir() / 'confusion_matrix.png'
            model.plot_confusion_matrix(
                eval_results['confusion_matrix'],
                save_path=cm_path
            )
        
        # 前処理器を保存
        preprocessor_path = exp_manager.get_experiment_dir() / 'preprocessors.pkl'
        data_loader.save_preprocessors(preprocessor_path)
        
        # 設定を保存
        config_path = exp_manager.get_experiment_dir() / 'config.yaml'
        config_manager.save_config(config, config_path)
        
        # 結果サマリーを保存
        summary = {
            'experiment_name': config.experiment_name,
            'cv_mean_f1': train_results['cv_mean'],
            'cv_std_f1': train_results['cv_std'],
            'val_f1_macro': eval_results['f1_macro'],
            'val_f1_weighted': eval_results['f1_weighted'],
            'feature_count': len(data_loader.get_feature_columns()),
            'train_samples': len(X_train),
            'val_samples': len(X_val)
        }
        
        summary_path = exp_manager.get_experiment_dir() / 'summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info("学習完了！")
        logger.info(f"CV平均F1スコア: {train_results['cv_mean']:.4f} ± {train_results['cv_std']:.4f}")
        logger.info(f"検証F1スコア: {eval_results['f1_macro']:.4f}")
        logger.info(f"実験ディレクトリ: {exp_manager.get_experiment_dir()}")
        
    except Exception as e:
        logger.error(f"学習中にエラーが発生しました: {e}")
        exp_manager.log_error(str(e))
        raise
    finally:
        exp_manager.end_experiment()


if __name__ == "__main__":
    main() 