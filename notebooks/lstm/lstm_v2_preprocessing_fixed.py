#!/usr/bin/env python
"""
LSTM v2前処理スクリプト（Demographics特徴量最適化対応）
Kernelクラッシュを避けるための単独実行版
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import PolynomialFeatures
import pickle
import os
import json
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# 設定
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")

# LSTM v2用のウィンドウ設定
WINDOW_CONFIGS = {
    'w64_s16': {'window_size': 64, 'stride': 16},
    'w128_s32': {'window_size': 128, 'stride': 32}
}

# Demographics特徴量最適化設定
ENABLE_DEMOGRAPHICS_OPTIMIZATION = True
DEMOGRAPHICS_OPTIMIZATION_METHOD = 'pca_18'  # 'original', 'feature_selection', 'pca_18', 'polynomial'

# 固定パラメータ
RANDOM_STATE = 42

print("LSTM v2 前処理パイプライン（最適化対応版）初期化完了")
print(f"設定済みウィンドウ設定: {list(WINDOW_CONFIGS.keys())}")
print(f"Demographics最適化: {DEMOGRAPHICS_OPTIMIZATION_METHOD if ENABLE_DEMOGRAPHICS_OPTIMIZATION else 'なし'}")

def optimize_demographics_features(X_demographics, method='pca_18', fit_transformer=True, transformer=None):
    """
    Demographics特徴量を最適化
    """
    if method == 'original':
        return X_demographics, None
    
    elif method == 'feature_selection':
        # 重要な特徴量のみ選択（上位8個）
        if fit_transformer:
            important_indices = [8, 6, 17, 2, 5, 7, 3, 16]  # 前回の最適化結果
            selected_features = X_demographics[:, important_indices]
            transformer = {'method': 'feature_selection', 'indices': important_indices}
        else:
            selected_features = X_demographics[:, transformer['indices']]
        return selected_features, transformer
    
    elif method == 'pca_18':
        # PCA 18次元（最良の結果を示した手法）
        if fit_transformer:
            pca = PCA(n_components=18, random_state=42)
            X_pca = pca.fit_transform(X_demographics)
            transformer = pca
        else:
            X_pca = transformer.transform(X_demographics)
        return X_pca, transformer
    
    elif method == 'polynomial':
        # 多項式特徴量（2次）+ 特徴量選択
        if fit_transformer:
            poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
            X_poly = poly.fit_transform(X_demographics)
            
            # 特徴量数が多い場合は上位50個に制限
            if X_poly.shape[1] > 50:
                selector = SelectKBest(score_func=f_classif, k=50)
                # ダミーラベルで仮フィット（実際の学習時は適切なラベルを使用）
                dummy_y = np.zeros(X_poly.shape[0])
                X_poly = selector.fit_transform(X_poly, dummy_y)
                transformer = {'poly': poly, 'selector': selector}
            else:
                transformer = {'poly': poly, 'selector': None}
        else:
            poly = transformer['poly']
            selector = transformer['selector']
            X_poly = poly.transform(X_demographics)
            if selector is not None:
                X_poly = selector.transform(X_poly)
        return X_poly, transformer
    
    else:
        raise ValueError(f"Unknown optimization method: {method}")

def create_demographics_features(demographics_df):
    """
    Demographics データから特徴量を作成
    """
    df = demographics_df.copy()
    
    # 年齢グループ
    df['age_group'] = pd.cut(df['age'], 
                            bins=[0, 12, 18, 25, 35, 100], 
                            labels=['child', 'teen', 'young_adult', 'adult', 'senior'])
    
    # 身長カテゴリ
    df['height_category'] = pd.cut(df['height_cm'], 
                                  bins=[0, 160, 170, 180, 200], 
                                  labels=['short', 'average', 'tall', 'very_tall'])
    
    # 腕の比率特徴量
    df['arm_ratio'] = df['elbow_to_wrist_cm'] / df['shoulder_to_wrist_cm']
    df['arm_length_relative'] = df['shoulder_to_wrist_cm'] / df['height_cm']
    
    # 性別×年齢の交互作用
    df['sex_age_interaction'] = df['sex'] * df['age']
    
    # 利き手×性別の交互作用
    df['handedness_sex'] = df['handedness'] * 2 + df['sex']
    
    # カテゴリカル変数をOne-Hot Encoding
    age_group_dummies = pd.get_dummies(df['age_group'], prefix='age_group')
    height_cat_dummies = pd.get_dummies(df['height_category'], prefix='height_cat')
    
    # 最終的な特徴量データフレーム
    features_df = pd.concat([
        df[['subject', 'adult_child', 'age', 'sex', 'handedness', 'height_cm', 
            'shoulder_to_wrist_cm', 'elbow_to_wrist_cm', 'arm_ratio', 
            'arm_length_relative', 'sex_age_interaction', 'handedness_sex']],
        age_group_dummies,
        height_cat_dummies
    ], axis=1)
    
    return features_df

def create_sliding_windows_with_demographics(df, window_size, stride, sensor_cols, demographics_cols, 
                                           min_sequence_length=10, padding_value=0.0):
    """
    Demographics情報を含むスライディングウィンドウを作成
    """
    X_sensor_windows = []
    X_demographics_windows = []
    y_windows = []
    sequence_info = []
    
    # 各シーケンス（被験者×ジェスチャー）ごとに処理
    grouped = df.groupby(['subject', 'sequence_id'])
    total_sequences = len(grouped)
    processed_sequences = 0
    skipped_sequences = 0
    padded_sequences = 0
    
    print(f"\n=== ウィンドウ作成統計 ===")
    
    for (subject, sequence_id), group in grouped:
        sequence_length = len(group)
        gesture = group['gesture'].iloc[0]
        
        # 短すぎるシーケンスはスキップ
        if sequence_length < min_sequence_length:
            skipped_sequences += 1
            continue
        
        processed_sequences += 1
        
        # センサーデータとdemographicsデータを取得
        sensor_data = group[sensor_cols].values
        demographics_data = group[demographics_cols].iloc[0].values  # 静的特徴量は同じ値
        
        # パディングが必要かチェック
        need_padding = sequence_length < window_size
        if need_padding:
            padded_sequences += 1
            # パディング
            padding_size = window_size - sequence_length
            padding = np.full((padding_size, len(sensor_cols)), padding_value)
            sensor_data = np.vstack([sensor_data, padding])
        
        # スライディングウィンドウ作成
        for start_idx in range(0, len(sensor_data) - window_size + 1, stride):
            end_idx = start_idx + window_size
            
            # ウィンドウデータを抽出
            sensor_window = sensor_data[start_idx:end_idx]
            
            X_sensor_windows.append(sensor_window)
            X_demographics_windows.append(demographics_data)
            y_windows.append(gesture)
            
            sequence_info.append({
                'subject': subject,
                'sequence_id': sequence_id,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'original_length': sequence_length,
                'padded': need_padding
            })
    
    # numpy配列に変換
    X_sensor_windows = np.array(X_sensor_windows, dtype=np.float32)
    X_demographics_windows = np.array(X_demographics_windows, dtype=np.float32)
    y_windows = np.array(y_windows)
    
    print(f"総シーケンス数: {total_sequences}")
    print(f"処理されたシーケンス数: {processed_sequences}")
    print(f"スキップされたシーケンス数: {skipped_sequences}")
    print(f"パディングされたシーケンス数: {padded_sequences}")
    print(f"総ウィンドウ数: {len(X_sensor_windows)}")
    print(f"センサーウィンドウ形状: {X_sensor_windows.shape}")
    print(f"Demographics特徴量形状: {X_demographics_windows.shape}")
    print(f"ラベル形状: {y_windows.shape}")
    
    return X_sensor_windows, X_demographics_windows, y_windows, sequence_info

def main():
    """
    メイン前処理関数
    """
    try:
        print("\n=== データ読み込み ===")
        # データの読み込み
        train_df = pd.read_csv('../data/train.csv')
        train_demographics = pd.read_csv('../data/train_demographics.csv')
        test_df = pd.read_csv('../data/test.csv')
        test_demographics = pd.read_csv('../data/test_demographics.csv')

        print(f"訓練データサイズ: {train_df.shape}")
        print(f"訓練Demographics サイズ: {train_demographics.shape}")
        print(f"テストデータサイズ: {test_df.shape}")
        print(f"テストDemographics サイズ: {test_demographics.shape}")

        # Demographics特徴量エンジニアリング
        print("\n=== Demographics特徴量エンジニアリング ===")
        train_demographics_features = create_demographics_features(train_demographics)
        test_demographics_features = create_demographics_features(test_demographics)

        print(f"訓練用特徴量数: {train_demographics_features.shape[1]}")
        print(f"テスト用特徴量数: {test_demographics_features.shape[1]}")

        # 時系列データとdemographics情報を結合
        print("\n=== データ結合 ===")
        train_with_demographics = train_df.merge(train_demographics_features, on='subject', how='left')
        test_with_demographics = test_df.merge(test_demographics_features, on='subject', how='left')

        print(f"結合後の訓練データサイズ: {train_with_demographics.shape}")
        print(f"結合後のテストデータサイズ: {test_with_demographics.shape}")

        # センサー列とdemographics列の特定
        sensor_cols = [col for col in train_df.columns if col.startswith(('acc_', 'rot_', 'tof_', 'thm_'))]
        demographics_feature_cols = [col for col in train_demographics_features.columns if col != 'subject']

        print(f"センサー列数: {len(sensor_cols)}")
        print(f"Demographics特徴量数: {len(demographics_feature_cols)}")

        # ラベルエンコーダーの準備
        print("\n=== ラベルエンコーディング ===")
        label_encoder = LabelEncoder()
        all_gestures = train_with_demographics['gesture'].unique()
        label_encoder.fit(all_gestures)
        print(f"ジェスチャー数: {len(all_gestures)}")

        # 各ウィンドウ設定で前処理を実行
        preprocessing_results = {}

        for config_name, config in WINDOW_CONFIGS.items():
            print(f"\n{'='*60}")
            print(f"前処理実行中: {config_name}")
            print(f"ウィンドウサイズ: {config['window_size']}, ストライド: {config['stride']}")
            print(f"{'='*60}")
            
            # スライディングウィンドウを作成
            X_sensor, X_demographics, y_windows, sequence_info = create_sliding_windows_with_demographics(
                train_with_demographics, 
                config['window_size'], 
                config['stride'], 
                sensor_cols, 
                demographics_feature_cols,
                min_sequence_length=10, 
                padding_value=0.0
            )
            
            # ラベルエンコーディング
            y_encoded = label_encoder.transform(y_windows)
            
            # センサーデータの正規化
            sensor_scaler = StandardScaler()
            n_samples, n_timesteps, n_sensor_features = X_sensor.shape
            X_sensor_flat = X_sensor.reshape(-1, n_sensor_features)
            
            # NaN値の処理
            nan_count = np.isnan(X_sensor_flat).sum()
            if nan_count > 0:
                print(f"  警告: センサーデータにNaN値が{nan_count}個見つかりました。0で置換します。")
                X_sensor_flat = np.nan_to_num(X_sensor_flat, nan=0.0)
            
            X_sensor_normalized = sensor_scaler.fit_transform(X_sensor_flat)
            X_sensor_normalized = X_sensor_normalized.reshape(n_samples, n_timesteps, n_sensor_features)
            
            # Demographics特徴量の最適化（正規化前）
            demographics_transformer = None
            if ENABLE_DEMOGRAPHICS_OPTIMIZATION and DEMOGRAPHICS_OPTIMIZATION_METHOD != 'original':
                print(f"\nDemographics特徴量最適化実行中...")
                print(f"  最適化前形状: {X_demographics.shape}")
                print(f"  最適化手法: {DEMOGRAPHICS_OPTIMIZATION_METHOD}")
                
                X_demographics_optimized, demographics_transformer = optimize_demographics_features(
                    X_demographics, 
                    method=DEMOGRAPHICS_OPTIMIZATION_METHOD, 
                    fit_transformer=True
                )
                
                print(f"  最適化後形状: {X_demographics_optimized.shape}")
                
                # 最適化後のDemographics特徴量を使用
                X_demographics = X_demographics_optimized
                demographics_feature_cols_optimized = [f"demo_opt_{i}" for i in range(X_demographics.shape[1])]
            else:
                print(f"\nDemographics特徴量最適化をスキップ（{DEMOGRAPHICS_OPTIMIZATION_METHOD}）")
                demographics_feature_cols_optimized = demographics_feature_cols
            
            # Demographics特徴量の正規化
            demographics_scaler = StandardScaler()
            X_demographics_normalized = demographics_scaler.fit_transform(X_demographics)
            
            print(f"\n正規化完了:")
            print(f"  センサーデータ: {X_sensor_normalized.shape}")
            print(f"  Demographics: {X_demographics_normalized.shape}")
            print(f"  ラベル: {y_encoded.shape}")
            
            # 結果を保存
            preprocessing_results[config_name] = {
                'X_sensor': X_sensor_normalized,
                'X_demographics': X_demographics_normalized,
                'y': y_encoded,
                'sequence_info': sequence_info,
                'sensor_scaler': sensor_scaler,
                'demographics_scaler': demographics_scaler,
                'demographics_transformer': demographics_transformer,
                'label_encoder': label_encoder,
                'config': config,
                'sensor_cols': sensor_cols,
                'demographics_cols': demographics_feature_cols_optimized,
                'optimization_method': DEMOGRAPHICS_OPTIMIZATION_METHOD if ENABLE_DEMOGRAPHICS_OPTIMIZATION else 'original'
            }
            
            print(f"\n{config_name} 前処理完了!")

        # データ保存
        print(f"\n{'='*60}")
        print("データ保存中...")
        print(f"{'='*60}")

        for config_name, result in preprocessing_results.items():
            print(f"\n=== {config_name} データ保存中 ===")
            
            # 出力ディレクトリの作成
            output_dir = Path(f"../output/experiments/lstm_v2_{config_name}")
            preprocessed_dir = output_dir / "preprocessed"
            preprocessed_dir.mkdir(parents=True, exist_ok=True)
            
            # データの保存
            with open(preprocessed_dir / "X_sensor_windows.pkl", "wb") as f:
                pickle.dump(result['X_sensor'], f)
            
            with open(preprocessed_dir / "X_demographics_windows.pkl", "wb") as f:
                pickle.dump(result['X_demographics'], f)
            
            with open(preprocessed_dir / "y_windows.pkl", "wb") as f:
                pickle.dump(result['y'], f)
            
            with open(preprocessed_dir / "sequence_info.pkl", "wb") as f:
                pickle.dump(result['sequence_info'], f)
            
            with open(preprocessed_dir / "sensor_scaler.pkl", "wb") as f:
                pickle.dump(result['sensor_scaler'], f)
            
            with open(preprocessed_dir / "demographics_scaler.pkl", "wb") as f:
                pickle.dump(result['demographics_scaler'], f)
            
            with open(preprocessed_dir / "label_encoder.pkl", "wb") as f:
                pickle.dump(result['label_encoder'], f)
            
            # Demographics最適化変換器の保存（存在する場合）
            if result.get('demographics_transformer') is not None:
                with open(preprocessed_dir / "demographics_transformer.pkl", "wb") as f:
                    pickle.dump(result['demographics_transformer'], f)
                print(f"  - 最適化変換器保存: demographics_transformer.pkl")
            
            # 設定とメタデータの保存
            config_data = {
                'window_size': result['config']['window_size'],
                'stride': result['config']['stride'],
                'n_sensor_features': len(result['sensor_cols']),
                'n_demographics_features': len(result['demographics_cols']),
                'n_classes': len(result['label_encoder'].classes_),
                'n_samples': len(result['y']),
                'sensor_cols': result['sensor_cols'],
                'demographics_cols': result['demographics_cols'],
                'gesture_classes': result['label_encoder'].classes_.tolist(),
                'demographics_optimization': result.get('optimization_method', 'original'),
                'demographics_optimized': result.get('optimization_method', 'original') != 'original'
            }
            
            with open(preprocessed_dir / "config.pkl", "wb") as f:
                pickle.dump(config_data, f)
            
            # JSON形式でも保存（可読性のため）
            with open(preprocessed_dir / "config.json", "w") as f:
                json.dump(config_data, f, indent=2)
            
            print(f"保存完了: {preprocessed_dir}")
            print(f"  - X_sensor_windows.pkl: {result['X_sensor'].shape}")
            print(f"  - X_demographics_windows.pkl: {result['X_demographics'].shape}")
            print(f"  - y_windows.pkl: {result['y'].shape}")
            print(f"  - 設定ファイル: config.pkl, config.json")
            print(f"  - 前処理器: sensor_scaler.pkl, demographics_scaler.pkl, label_encoder.pkl")
            
            # 最適化情報の表示
            optimization_method = result.get('optimization_method', 'original')
            print(f"  - Demographics最適化: {optimization_method}")

        print(f"\n{'='*60}")
        print("LSTM v2 前処理パイプライン完了！")
        print(f"Demographics最適化: {DEMOGRAPHICS_OPTIMIZATION_METHOD if ENABLE_DEMOGRAPHICS_OPTIMIZATION else 'なし'}")
        print("\n次のステップ:")
        print("1. LSTM v2モデルの実装（時系列 + demographics入力）")
        print("2. Attention機構の追加")
        print("3. ハイパーパラメータ最適化（最適化済み特徴量で効率化）")
        print("4. Transformerベースモデルの実装")
        print(f"\n保存されたデータ:")
        for config_name in preprocessing_results.keys():
            print(f"  - ../output/experiments/lstm_v2_{config_name}/preprocessed/")

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 