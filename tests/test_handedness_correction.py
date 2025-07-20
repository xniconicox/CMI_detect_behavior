#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""利き手補正関数のテストスクリプト"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent))

from src.utils.preprocessing import handedness_correction_v2, handedness_correction_v2_reverse, augment_by_handedness_flip


def test_handedness_correction():
    """利き手補正関数のテスト"""
    print("=== 利き手補正関数のテスト ===")
    
    # テストデータ作成
    test_data = pd.DataFrame({
        'subject': [1, 1, 2, 2],
        'handedness': [0, 1, 0, 1],  # 0: 左利き, 1: 右利き
        'acc_x': [1.0, 2.0, 3.0, 4.0],
        'acc_y': [1.0, 2.0, 3.0, 4.0],
        'acc_z': [1.0, 2.0, 3.0, 4.0],
        'rot_w': [1.0, 1.0, 1.0, 1.0],
        'rot_x': [0.1, 0.2, 0.3, 0.4],
        'rot_y': [0.1, 0.2, 0.3, 0.4],
        'rot_z': [0.1, 0.2, 0.3, 0.4],
    })
    
    print("元データ:")
    print(test_data)
    print()
    
    # 左利き→右利き変換
    corrected_left = handedness_correction_v2(test_data.copy())
    print("左利き→右利き変換後:")
    print(corrected_left)
    print()
    
    # 右利き→左利き変換
    corrected_right = handedness_correction_v2_reverse(test_data.copy())
    print("右利き→左利き変換後:")
    print(corrected_right)
    print()
    
    # 変換の検証
    print("=== 変換検証 ===")
    print("左利きデータ (handedness=0) の変換:")
    left_data = test_data[test_data['handedness'] == 0]
    corrected_left_data = corrected_left[corrected_left['handedness'] == 0]
    
    print("元の左利き acc_y:", left_data['acc_y'].values)
    print("変換後 acc_y:", corrected_left_data['acc_y'].values)
    print("期待値: 符号反転 (-1.0, -3.0)")
    print()
    
    print("右利きデータ (handedness=1) の変換:")
    right_data = test_data[test_data['handedness'] == 1]
    corrected_right_data = corrected_right[corrected_right['handedness'] == 1]
    
    print("元の右利き acc_y:", right_data['acc_y'].values)
    print("変換後 acc_y:", corrected_right_data['acc_y'].values)
    print("期待値: 符号反転 (-2.0, -4.0)")
    print()


def test_augmentation():
    """データ拡張関数のテスト"""
    print("=== データ拡張関数のテスト ===")
    
    # テストデータ作成
    test_data = pd.DataFrame({
        'subject': [1, 2],
        'handedness': [0, 1],  # 左利き、右利き
        'acc_x': [1.0, 2.0],
        'acc_y': [1.0, 2.0],
        'acc_z': [1.0, 2.0],
    })
    
    print("元データ:")
    print(test_data)
    print()
    
    # データ拡張実行
    augmented_data = augment_by_handedness_flip(test_data)
    print("拡張後データ:")
    print(augmented_data)
    print()
    
    # 結果検証
    print("=== 拡張結果検証 ===")
    print(f"元データ数: {len(test_data)}")
    print(f"拡張後データ数: {len(augmented_data)}")
    print(f"期待値: {len(test_data) * 2}")
    
    # 新しいsubject IDの確認
    original_subjects = set(test_data['subject'])
    new_subjects = set(augmented_data['subject']) - original_subjects
    print(f"元のsubject: {original_subjects}")
    print(f"新しいsubject: {new_subjects}")
    
    # handednessの分布確認
    print("\nhandedness分布:")
    print(augmented_data['handedness'].value_counts().sort_index())


if __name__ == "__main__":
    test_handedness_correction()
    print("\n" + "="*50 + "\n")
    test_augmentation() 