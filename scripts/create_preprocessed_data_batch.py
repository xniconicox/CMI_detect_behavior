#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
メモリ効率的なバッチ処理による前処理スクリプト
パディング付きスライディングウィンドウでデータ損失を防ぐ
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os
from pathlib import Path
import warnings
import gc
warnings.filterwarnings('ignore')

def create_sliding_windows_batch(df, window_size, stride, sensor_cols, batch_size=1000, min_sequence_length=10, padding_value=0.0):
    """
    バッチ処理でスライディングウィンドウを作成
    """
    # 出力ディレクトリを準備
    temp_dir = Path("temp_windows")
    temp_dir.mkdir(exist_ok=True)
    
    sequence_ids = df['sequence_id'].unique()
    total_sequences = len(sequence_ids)
    
    # 統計情報
    padded_sequences = 0
    skipped_sequences = 0
    total_windows = 0
    batch_count = 0
    
    # ラベルエンコーダーを準備
    label_encoder = LabelEncoder()
    all_gestures = df['gesture'].unique()
    label_encoder.fit(all_gestures)
    
    print(f"総シーケンス数: {total_sequences}")
    print(f"バッチサイズ: {batch_size}")
    print(f"予想バッチ数: {(total_sequences + batch_size - 1) // batch_size}")
    
    # バッチごとに処理
    for batch_start in range(0, total_sequences, batch_size):
        batch_end = min(batch_start + batch_size, total_sequences)
        batch_seq_ids = sequence_ids[batch_start:batch_end]
        
        print(f"\nバッチ {batch_count + 1}: シーケンス {batch_start+1}-{batch_end}")
        
        X_batch = []
        y_batch = []
        info_batch = []
        
        # バッチ内のシーケンスを処理
        for seq_id in batch_seq_ids:
            seq_data = df[df['sequence_id'] == seq_id].copy()
            seq_data = seq_data.sort_values('sequence_counter')
            
            if len(seq_data) < min_sequence_length:
                skipped_sequences += 1
                continue
            
            sensor_data = seq_data[sensor_cols].values
            gesture = seq_data['gesture'].iloc[0]
            original_length = len(sensor_data)
            
            # パディング処理
            is_padded = False
            if len(sensor_data) < window_size:
                padding_length = window_size - len(sensor_data)
                padding = np.full((padding_length, len(sensor_cols)), padding_value)
                sensor_data = np.vstack([sensor_data, padding])
                is_padded = True
                padded_sequences += 1
            
            # スライディングウィンドウ
            for i in range(0, len(sensor_data) - window_size + 1, stride):
                window = sensor_data[i:i + window_size]
                X_batch.append(window)
                y_batch.append(gesture)
                info_batch.append({
                    'sequence_id': seq_id,
                    'start_idx': i,
                    'gesture': gesture,
                    'original_length': original_length,
                    'is_padded': is_padded,
                    'batch_id': batch_count
                })
                total_windows += 1
        
        # バッチデータを保存
        if X_batch:
            X_batch_array = np.array(X_batch)
            y_batch_array = label_encoder.transform(y_batch)
            
            # NaN/無限大処理
            X_flat = X_batch_array.reshape(-1, X_batch_array.shape[-1])
            nan_count = np.isnan(X_flat).sum()
            if nan_count > 0:
                print(f"  NaN値 {nan_count} 個を0で置換")
                X_flat = np.nan_to_num(X_flat, nan=0.0, posinf=1.0, neginf=-1.0)
                X_batch_array = X_flat.reshape(X_batch_array.shape)
            
            # バッチファイルを保存
            with open(temp_dir / f"X_batch_{batch_count:03d}.pkl", "wb") as f:
                pickle.dump(X_batch_array, f)
            with open(temp_dir / f"y_batch_{batch_count:03d}.pkl", "wb") as f:
                pickle.dump(y_batch_array, f)
            with open(temp_dir / f"info_batch_{batch_count:03d}.pkl", "wb") as f:
                pickle.dump(info_batch, f)
            
            print(f"  バッチ保存: {X_batch_array.shape}")
            
            # メモリクリア
            del X_batch, y_batch, X_batch_array, y_batch_array
            gc.collect()
        
        batch_count += 1
    
    print(f"\n=== バッチ処理完了 ===")
    print(f"総ウィンドウ数: {total_windows}")
    print(f"パディング済み: {padded_sequences}")
    print(f"スキップ: {skipped_sequences}")
    print(f"保存バッチ数: {batch_count}")
    
    return batch_count, total_windows, label_encoder

def combine_batches_and_normalize(batch_count, config_name, target_dir):
    """
    バッチファイルを結合して正規化
    """
    print(f"\n=== バッチ結合・正規化開始 ===")
    
    temp_dir = Path("temp_windows")
    output_dir = Path(target_dir)
    preprocessed_dir = output_dir / "preprocessed"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    
    # 正規化用の統計量を計算
    print("正規化統計量を計算中...")
    scaler = StandardScaler()
    
    # 各バッチからサンプルを取得して統計量を計算
    sample_data = []
    for i in range(min(5, batch_count)):  # 最初の5バッチから統計量を計算
        try:
            with open(temp_dir / f"X_batch_{i:03d}.pkl", "rb") as f:
                X_batch = pickle.load(f)
                # 各バッチから一部をサンプリング
                sample_size = min(100, len(X_batch))
                indices = np.random.choice(len(X_batch), sample_size, replace=False)
                sample_data.append(X_batch[indices])
        except FileNotFoundError:
            continue
    
    if sample_data:
        sample_combined = np.vstack(sample_data)
        sample_flat = sample_combined.reshape(-1, sample_combined.shape[-1])
        scaler.fit(sample_flat)
        print(f"統計量計算完了: {sample_flat.shape}")
        del sample_data, sample_combined, sample_flat
        gc.collect()
    
    # バッチを順次読み込み、正規化して結合
    print("バッチを正規化・結合中...")
    all_X = []
    all_y = []
    all_info = []
    
    for i in range(batch_count):
        try:
            # バッチ読み込み
            with open(temp_dir / f"X_batch_{i:03d}.pkl", "rb") as f:
                X_batch = pickle.load(f)
            with open(temp_dir / f"y_batch_{i:03d}.pkl", "rb") as f:
                y_batch = pickle.load(f)
            with open(temp_dir / f"info_batch_{i:03d}.pkl", "rb") as f:
                info_batch = pickle.load(f)
            
            # 正規化
            original_shape = X_batch.shape
            X_flat = X_batch.reshape(-1, original_shape[-1])
            X_scaled_flat = scaler.transform(X_flat)
            X_scaled = X_scaled_flat.reshape(original_shape)
            
            all_X.append(X_scaled)
            all_y.append(y_batch)
            all_info.extend(info_batch)
            
            print(f"  バッチ {i+1}/{batch_count} 処理完了: {X_scaled.shape}")
            
        except FileNotFoundError:
            print(f"  バッチ {i} が見つかりません")
            continue
    
    # 最終結合
    print("最終結合中...")
    X_final = np.vstack(all_X)
    y_final = np.hstack(all_y)
    
    print(f"最終データ形状: X={X_final.shape}, y={y_final.shape}")
    
    # 保存
    print("データ保存中...")
    with open(preprocessed_dir / "X_windows.pkl", "wb") as f:
        pickle.dump(X_final, f)
    with open(preprocessed_dir / "y_windows.pkl", "wb") as f:
        pickle.dump(y_final, f)
    with open(preprocessed_dir / "sequence_info.pkl", "wb") as f:
        pickle.dump(all_info, f)
    with open(preprocessed_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    print(f"保存完了: {preprocessed_dir}")
    
    # 一時ファイルを削除
    print("一時ファイル削除中...")
    for i in range(batch_count):
        for prefix in ["X_batch", "y_batch", "info_batch"]:
            file_path = temp_dir / f"{prefix}_{i:03d}.pkl"
            if file_path.exists():
                file_path.unlink()
    
    if temp_dir.exists() and not any(temp_dir.iterdir()):
        temp_dir.rmdir()
    
    return X_final.shape, y_final.shape

def main():
    """メイン処理"""
    print("=== メモリ効率的前処理開始 ===")
    
    # 設定
    config = {'window_size': 128, 'stride': 32}
    config_name = 'w128_s32'
    batch_size = 500  # メモリに応じて調整
    
    # データ読み込み
    print("データ読み込み中...")
    train_df = pd.read_csv('data/train.csv')
    sensor_cols = [col for col in train_df.columns if col.startswith(('acc_', 'rot_', 'tof_', 'thm_'))]
    
    print(f"データ形状: {train_df.shape}")
    print(f"センサー列数: {len(sensor_cols)}")
    
    # バッチ処理でウィンドウ作成
    batch_count, total_windows, label_encoder = create_sliding_windows_batch(
        train_df, config['window_size'], config['stride'], sensor_cols, batch_size
    )
    
    # バッチ結合・正規化
    target_dir = f"output/experiments/lstm_{config_name}"
    X_shape, y_shape = combine_batches_and_normalize(batch_count, config_name, target_dir)
    
    # 設定とラベルエンコーダーを保存
    preprocessed_dir = Path(target_dir) / "preprocessed"
    with open(preprocessed_dir / "config.pkl", "wb") as f:
        pickle.dump(config, f)
    with open(preprocessed_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
    
    print(f"\n=== 処理完了 ===")
    print(f"最終データ形状: X={X_shape}, y={y_shape}")
    print(f"出力ディレクトリ: {target_dir}")
    print(f"改善: 922 → {X_shape[0]} サンプル ({X_shape[0]/922:.1f}倍)")

if __name__ == "__main__":
    main() 