#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.csvを訓練時と同じ分割で分割して、評価用データtrain_val.csvを作成するスクリプト
対応するdemographicsも評価用のcsvを用意する
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import argparse

def create_train_val_split(train_csv_path, train_demographics_csv_path, 
                          output_dir, test_size=0.2, val_size=0.2, random_state=42):
    """
    train.csvを訓練時と同じ分割で分割して評価用データを作成
    sequence_idごとにsequence_counter順で結合し、subjectリストをsplitして分割
    """
    print("=" * 60)
    print("訓練時と同じ分割で評価用データを作成中... (subject単位)")
    print("=" * 60)
    
    # 出力ディレクトリ作成
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # データ読み込み
    print(f"📊 train.csv読み込み中: {train_csv_path}")
    train_df = pd.read_csv(train_csv_path)
    print(f"  形状: {train_df.shape}")
    print(f"  列数: {len(train_df.columns)}")
    
    print(f"📊 train_demographics.csv読み込み中: {train_demographics_csv_path}")
    train_demo_df = pd.read_csv(train_demographics_csv_path)
    print(f"  形状: {train_demo_df.shape}")
    print(f"  列数: {len(train_demo_df.columns)}")

    # sequence_idごとにsequence_counter順で結合
    print("\n🔗 sequence_idごとにsequence_counter順で結合...")
    grouped = train_df.sort_values(by=["sequence_id", "sequence_counter"]).groupby("sequence_id")
    sequence_list = [g for _, g in grouped]
    sequence_subjects = [g["subject"].iloc[0] for g in sequence_list]
    sequence_ids = [g["sequence_id"].iloc[0] for g in sequence_list]

    # subjectリストを一意に抽出
    unique_subjects = sorted(train_df["subject"].unique())
    print(f"ユニークsubject数: {len(unique_subjects)}")

    # subject単位でsplit（9:1）
    train_subjects, val_subjects = train_test_split(
        unique_subjects, test_size=0.01, random_state=random_state
    )
    print(f"train_subjects: {len(train_subjects)} / val_subjects: {len(val_subjects)}")

    # subject→sequence_idの対応
    subject_to_seqids = {}
    for seq_id, subj in zip(sequence_ids, sequence_subjects):
        subject_to_seqids.setdefault(subj, []).append(seq_id)

    # 各分割のsequence_idリスト
    train_seqids = sum([subject_to_seqids[s] for s in train_subjects if s in subject_to_seqids], [])
    val_seqids = sum([subject_to_seqids[s] for s in val_subjects if s in subject_to_seqids], [])

    # 各分割のデータ抽出
    train_df_split = train_df[train_df["sequence_id"].isin(train_seqids)].sort_values(by=["subject", "sequence_id", "sequence_counter"])
    val_df_split = train_df[train_df["sequence_id"].isin(val_seqids)].sort_values(by=["subject", "sequence_id", "sequence_counter"])

    # 元のカラム順・型に揃える
    train_df_split = train_df_split[train_df.columns].astype(train_df.dtypes.to_dict())
    val_df_split = val_df_split[train_df.columns].astype(train_df.dtypes.to_dict())

    # インデックスをリセット（index=Falseで保存するため）
    train_df_split = train_df_split.reset_index(drop=True)
    val_df_split = val_df_split.reset_index(drop=True)

    # Demographicsデータの分割
    train_demo_split = train_demo_df[train_demo_df["subject"].isin(train_subjects)].reset_index(drop=True)
    val_demo_split = train_demo_df[train_demo_df["subject"].isin(val_subjects)].reset_index(drop=True)

    print(f"\n📈 分割結果詳細:")
    print(f"  訓練データ: センサーデータ: {train_df_split.shape} 人口統計データ: {train_demo_split.shape} ユニーク被験者数: {len(train_subjects)}")
    print(f"  検証データ: センサーデータ: {val_df_split.shape} 人口統計データ: {val_demo_split.shape} ユニーク被験者数: {len(val_subjects)}")

    # ファイル保存
    print(f"\n💾 ファイル保存中...")
    train_output_path = output_path / "train.csv"
    train_df_split.to_csv(train_output_path, index=False)
    print(f"  訓練データ: {train_output_path}")
    train_demo_output_path = output_path / "train_demographics.csv"
    train_demo_split.to_csv(train_demo_output_path, index=False)
    print(f"  訓練人口統計: {train_demo_output_path}")
    val_output_path = output_path / "train_val.csv"
    val_df_split.to_csv(val_output_path, index=False)
    print(f"  評価用データ: {val_output_path}")
    val_demo_output_path = output_path / "train_val_demographics.csv"
    val_demo_split.to_csv(val_demo_output_path, index=False)
    print(f"  評価用人口統計: {val_demo_output_path}")

    # test.csv = train_val.csvから不要な列を削除したもの
    test_output_path = output_path / "test.csv"
    drop_cols = [col for col in ["gesture", "sequence_type", "orientation", "behavior", "phase"] if col in val_df_split.columns]
    test_df = val_df_split.drop(columns=drop_cols) if drop_cols else val_df_split.copy()
    test_df.to_csv(test_output_path, index=False)
    print(f"  テストデータ: {test_output_path}")
    # test_demographics.csv = train_val_demographics.csvの複製
    test_demo_output_path = output_path / "test_demographics.csv"
    val_demo_split.to_csv(test_demo_output_path, index=False)
    print(f"  テスト人口統計: {test_demo_output_path}")

    # test_labels.csv = val_df_splitからsequence_idとgestureのみ抽出
    if "gesture" in val_df_split.columns:
        val_df_split = pd.DataFrame(val_df_split)
        test_labels_output_path = output_path / "test_labels.csv"
        test_labels_df = val_df_split[["sequence_id", "gesture"]].drop_duplicates().reset_index(drop=True)
        test_labels_df.to_csv(test_labels_output_path, index=False)
        print(f"  テスト正解ラベル: {test_labels_output_path}")

    # 分割情報の保存
    split_info = {
        'split_config': {
            'val_size': 0.1,
            'random_state': random_state
        },
        'subject_counts': {
            'train': len(train_subjects),
            'val': len(val_subjects),
            'total': len(unique_subjects)
        },
        'file_paths': {
            'train_csv': str(train_output_path),
            'train_demographics_csv': str(train_demo_output_path),
            'train_val_csv': str(val_output_path),
            'train_val_demographics_csv': str(val_demo_output_path),
            'test_csv': str(test_output_path),
            'test_demographics_csv': str(test_demo_output_path)
        }
    }
    import json
    split_info_path = output_path / "split_info.json"
    with open(split_info_path, 'w', encoding='utf-8') as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)
    print(f"  分割情報: {split_info_path}")
    print(f"\n🎉 評価用データ作成完了!")
    print(f"📁 出力ディレクトリ: {output_path}")
    print(f"📄 評価用データ: train_val.csv")
    print(f"📄 評価用人口統計: train_val_demographics.csv")
    print(f"📄 テストデータ: test.csv（gesture列なし）")
    print(f"📄 テスト人口統計: test_demographics.csv")

def main():
    parser = argparse.ArgumentParser(description='train.csvを訓練時と同じ分割で分割して評価用データを作成 (subject単位)')
    parser.add_argument('--train-csv', type=str, default='data/train.csv',
                       help='train.csvのパス (デフォルト: data/train.csv)')
    parser.add_argument('--train-demographics-csv', type=str, default='data/train_demographics.csv',
                       help='train_demographics.csvのパス (デフォルト: data/train_demographics.csv)')
    parser.add_argument('--output-dir', type=str, default='submissions/data/split',
                       help='出力ディレクトリ (デフォルト: submissions/data/split)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='テストデータの割合 (デフォルト: 0.2)')
    parser.add_argument('--val-size', type=float, default=0.2,
                       help='検証データの割合 (デフォルト: 0.2)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='乱数シード (デフォルト: 42)')
    args = parser.parse_args()
    if not Path(args.train_csv).exists():
        print(f"❌ エラー: train.csvが見つかりません: {args.train_csv}")
        return
    if not Path(args.train_demographics_csv).exists():
        print(f"❌ エラー: train_demographics.csvが見つかりません: {args.train_demographics_csv}")
        return
    create_train_val_split(
        train_csv_path=args.train_csv,
        train_demographics_csv_path=args.train_demographics_csv,
        output_dir=args.output_dir,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state
    )

if __name__ == "__main__":
    main()

    
'''
    # デフォルト設定で実行
python scripts/create_train_val_split.py

# カスタム設定で実行
python scripts/create_train_val_split.py \
    --train-csv data/train.csv \
    --train-demographics-csv data/train_demographics.csv \
    --output-dir data/split \
    --test-size 0.2 \
    --val-size 0.2 \
    --random-state 42
'''