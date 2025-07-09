#!/usr/bin/env python3
"""
提出用ディレクトリ更新スクリプト
最新の学習結果を使用して提出用ファイルを更新
"""

import os
import json
import shutil
from pathlib import Path
import glob
from datetime import datetime

def find_latest_results():
    """最新の学習結果を検索"""
    results_dir = Path("/mnt/c/Users/ShunK/works/CMI_comp/results/lstm_v2")
    
    # 結果ファイルのパターンを検索
    result_files = glob.glob(str(results_dir / "final_results_*.json"))
    
    if not result_files:
        raise FileNotFoundError("学習結果ファイルが見つかりません")
    
    # 最新のファイルを取得（タイムスタンプ順）
    latest_file = max(result_files, key=os.path.getmtime)
    
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    print(f"最新の学習結果: {latest_file}")
    print(f"CMIスコア: {results['final_cmi_score']:.4f}")
    print(f"タイムスタンプ: {results['timestamp']}")
    
    return results

def update_submission_files(results):
    """提出用ファイルを更新"""
    submission_dir = Path("/mnt/c/Users/ShunK/works/CMI_comp/submissions/lstm_v2")
    timestamp = results['timestamp']
    
    print(f"\n📁 提出用ディレクトリ更新中: {submission_dir}")
    
    # 1. モデルファイルをコピー
    model_files_to_copy = [
        ("model_path", f"final_model_{timestamp}.keras")
    ]
    
    # 重みファイルのパスを修正
    weights_path = f"/mnt/c/Users/ShunK/works/CMI_comp/results/lstm_v2/checkpoints/final_model_{timestamp}.weights.h5"
    if os.path.exists(weights_path):
        model_files_to_copy.append(("weights_file", f"final_model_{timestamp}.weights.h5"))
        results["weights_file"] = weights_path
    
    for key, filename in model_files_to_copy:
        if key in results:
            source_path = Path(results[key])
            if source_path.exists():
                dest_path = submission_dir / filename
                shutil.copy2(source_path, dest_path)
                print(f"✅ コピー完了: {filename}")
            else:
                print(f"⚠️ ファイルが見つかりません: {source_path}")
    
    # 2. 設定ファイルを作成
    config_data = {
        "model_info": {
            "timestamp": timestamp,
            "cmi_score": results['final_cmi_score'],
            "binary_f1": results['binary_f1'],
            "macro_f1": results['macro_f1'],
            "test_accuracy": results['test_accuracy'],
            "window_config": results['window_config'],
            "epochs_trained": results['epochs_trained']
        },
        "model_params": results['model_params'],
        "best_params": results['best_params'],
        "file_paths": {
            "model_file": f"final_model_{timestamp}.keras",
            "weights_file": f"final_model_{timestamp}.weights.h5",
            "architecture_file": f"final_model_{timestamp}_architecture.json"
        }
    }
    
    config_path = submission_dir / "model_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    print(f"✅ 設定ファイル作成: model_config.json")
    
    # 3. アーキテクチャファイルをコピー
    # .kerasファイルから.jsonファイルを探す
    model_path = results['model_path']
    if model_path.endswith('.keras'):
        architecture_source = Path(model_path.replace('.keras', '_architecture.json'))
    else:
        architecture_source = Path(model_path.replace('.h5', '_architecture.json'))
    
    if architecture_source.exists():
        architecture_dest = submission_dir / f"final_model_{timestamp}_architecture.json"
        shutil.copy2(architecture_source, architecture_dest)
        print(f"✅ アーキテクチャファイルコピー: {architecture_dest.name}")
    else:
        print(f"⚠️ アーキテクチャファイルが見つかりません: {architecture_source}")
    
    # 4. READMEを更新
    update_readme(submission_dir, results)
    
    # 5. 推論スクリプトを更新
    update_inference_script(submission_dir, results)
    
    print(f"\n🎉 提出用ディレクトリ更新完了!")
    return config_data

def update_readme(submission_dir, results):
    """READMEファイルを更新"""
    readme_path = submission_dir / "README.md"
    timestamp = results['timestamp']
    
    readme_content = f"""# CMI 2025 LSTM v2 Submission

## モデル情報
- **タイムスタンプ**: {timestamp}
- **CMIスコア**: {results['final_cmi_score']:.4f}
- **Binary F1**: {results['binary_f1']:.4f}
- **Macro F1**: {results['macro_f1']:.4f}
- **テスト精度**: {results['test_accuracy']:.4f}
- **学習エポック数**: {results['epochs_trained']}
- **ウィンドウ設定**: {results['window_config']}

## モデルアーキテクチャ
- **融合方式**: {results['model_params']['fusion_type']}
- **LSTM Units**: {results['model_params']['lstm_units_1']} → {results['model_params']['lstm_units_2']}
- **Dense Units**: {results['model_params']['dense_units']}
- **Demographics Dense Units**: {results['model_params']['demographics_dense_units']}
- **Fusion Dense Units**: {results['model_params']['fusion_dense_units']}
- **Dropout Rate**: {results['model_params']['dropout_rate']}
- **Learning Rate**: {results['model_params']['learning_rate']}
- **Batch Size**: {results['model_params']['batch_size']}

## ファイル構成
- `final_model_{timestamp}.keras` - メインモデルファイル
- `final_model_{timestamp}.weights.h5` - モデル重み
- `final_model_{timestamp}_architecture.json` - モデルアーキテクチャ
- `model_config.json` - 設定情報
- `model_inference.py` - 推論スクリプト
- `submit_final_model.py` - 提出用スクリプト

## 使用方法

### 推論実行
```bash
python model_inference.py --input_data path/to/test_data.csv --output predictions.csv
```

### 提出用ファイル生成
```bash
python submit_final_model.py
```

## 性能詳細
- 最適化時スコア: {results['optimization_score']:.4f}
- 学習完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 注意事項
- このモデルは150エポックで学習されました
- 最適なパラメータは Optuna による最適化結果を使用
- GPU環境での学習を推奨します
"""
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"✅ README更新完了")

def update_inference_script(submission_dir, results):
    """推論スクリプトを更新"""
    inference_script = submission_dir / "model_inference.py"
    timestamp = results['timestamp']
    
    # 既存のスクリプトを読み込み
    if inference_script.exists():
        with open(inference_script, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # モデルファイル名を更新
        content = content.replace(
            'final_model_', 
            f'final_model_{timestamp}'
        )
        
        # 設定を更新
        model_config_section = f'''
# モデル設定（自動生成）
MODEL_CONFIG = {{
    "timestamp": "{timestamp}",
    "cmi_score": {results['final_cmi_score']},
    "model_file": "final_model_{timestamp}.keras",
    "weights_file": "final_model_{timestamp}.weights.h5",
    "window_config": "{results['window_config']}",
    "model_params": {json.dumps(results['model_params'], indent=4)}
}}
'''
        
        # 既存の設定セクションを置換
        import re
        pattern = r'# モデル設定.*?^MODEL_CONFIG = \{.*?\}$'
        if re.search(pattern, content, re.MULTILINE | re.DOTALL):
            content = re.sub(pattern, model_config_section.strip(), content, flags=re.MULTILINE | re.DOTALL)
        else:
            # 設定セクションが見つからない場合は先頭に追加
            content = model_config_section + '\n\n' + content
        
        with open(inference_script, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ 推論スクリプト更新完了")

def main():
    """メイン実行関数"""
    print("🚀 提出用ディレクトリ更新開始")
    print("=" * 60)
    
    try:
        # 最新の学習結果を検索
        results = find_latest_results()
        
        # 提出用ファイルを更新
        config_data = update_submission_files(results)
        
        print("\n📊 更新サマリー:")
        print(f"  - CMIスコア: {results['final_cmi_score']:.4f}")
        print(f"  - タイムスタンプ: {results['timestamp']}")
        print(f"  - モデルファイル: final_model_{results['timestamp']}.keras")
        print(f"  - 重みファイル: final_model_{results['timestamp']}.weights.h5")
        
        print(f"\n✅ 提出用ディレクトリ更新完了!")
        print(f"📁 提出ディレクトリ: /mnt/c/Users/ShunK/works/CMI_comp/submissions/lstm_v2")
        
    except Exception as e:
        print(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 