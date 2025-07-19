#!/usr/bin/env python3
"""
CMI評価関数のテストスクリプト
"""

import numpy as np
import sys
import os

# プロジェクトルートを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.cmi_evaluation import calculate_cmi_score

def test_cmi_evaluation():
    """CMI評価関数のテスト"""
    print("🧪 CMI評価関数のテスト開始")
    
    # テストデータ作成（マルチクラス分類）
    np.random.seed(42)
    n_samples = 100
    n_classes = 5
    
    # 真のラベルと予測ラベル
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = np.random.randint(0, n_classes, n_samples)
    
    print(f"テストデータ: {n_samples}サンプル, {n_classes}クラス")
    print(f"y_true: {y_true[:10]}...")
    print(f"y_pred: {y_pred[:10]}...")
    
    try:
        # CMI評価指標計算（ラベルエンコーダーなし）
        cmi_score, binary_f1, macro_f1, accuracy = calculate_cmi_score(
            y_pred, y_true, label_encoder=None, verbose=True
        )
        
        print(f"\n✅ テスト成功!")
        print(f"CMI Score: {cmi_score:.4f}")
        print(f"Binary F1: {binary_f1:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_cmi_evaluation()
    if success:
        print("\n🎉 CMI評価関数は正常に動作しています")
    else:
        print("\n💥 CMI評価関数に問題があります") 