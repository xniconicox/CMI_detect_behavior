#!/usr/bin/env python3
"""
CMI Competition Evaluation Module

CMIコンペティション用の評価指標を計算するモジュール
"""

import numpy as np
from sklearn.metrics import f1_score, accuracy_score

# ターゲットジェスチャー（刺激行動）
TARGET_GESTURES = [
    'Above ear - pull hair',
    'Cheek - pinch skin', 
    'Eyebrow - pull hair',
    'Eyelash - pull hair',
    'Forehead - pull hairline',
    'Forehead - scratch',
    'Neck - pinch skin',
    'Neck - scratch'
]

# 非ターゲットジェスチャー（その他の行動）
NON_TARGET_GESTURES = [
    'Write name on leg',
    'Wave hello', 
    'Glasses on/off',
    'Text on phone',
    'Write name in air',
    'Feel around in tray and pull out an object',
    'Scratch knee/leg skin',
    'Pull air toward your face',
    'Drink from bottle/cup',
    'Pinch knee/leg skin'
]

def calculate_cmi_score(y_pred, y_true, label_encoder=None, verbose=False):
    """
    CMI コンペティション評価指標の計算
    
    Parameters:
    -----------
    y_pred : array-like
        予測ラベル（エンコード済み）
    y_true : array-like
        真のラベル（エンコード済み）
    label_encoder : LabelEncoder, optional
        ラベルエンコーダー（None の場合は数値ラベルを直接使用）
    verbose : bool, default=False
        詳細ログの出力フラグ
    
    Returns:
    --------
    tuple
        (CMI スコア, Binary F1, Macro F1, Test Accuracy)
    """
    try:
        if verbose:
            print(f"CMI評価指標計算開始...")
            print(f"y_true shape: {np.array(y_true).shape}, y_pred shape: {np.array(y_pred).shape}")
        
        # numpy配列に変換
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Test Accuracy計算
        test_accuracy = accuracy_score(y_true, y_pred)
        
        if label_encoder is not None:
            # ラベルを元の文字列に変換
            y_true_str = label_encoder.inverse_transform(y_true)
            y_pred_str = label_encoder.inverse_transform(y_pred)
            
            if verbose:
                print(f"ラベル変換完了: {len(y_true_str)} samples")
                unique_gestures = np.unique(np.concatenate([y_true_str, y_pred_str]))
                print(f"データ中のジェスチャー: {len(unique_gestures)}種類")
            
            # 1. Binary F1: Target vs Non-Target
            y_true_binary = np.array([1 if gesture in TARGET_GESTURES else 0 for gesture in y_true_str])
            y_pred_binary = np.array([1 if gesture in TARGET_GESTURES else 0 for gesture in y_pred_str])
            
            if verbose:
                print(f"Binary分類 - Target: {np.sum(y_true_binary)}, Non-Target: {np.sum(1-y_true_binary)}")
            
            # Zero division回避
            if len(np.unique(y_true_binary)) == 1 or len(np.unique(y_pred_binary)) == 1:
                if verbose:
                    print("Binary分類で単一クラスのみ検出 - F1スコアを0に設定")
                binary_f1 = 0.0
            else:
                binary_f1 = f1_score(y_true_binary, y_pred_binary, average='binary', zero_division='warn')
            
            # 2. Macro F1: 全ジェスチャーのマクロF1（非ターゲットは単一クラスに統合）
            y_true_macro = np.array([gesture if gesture in TARGET_GESTURES else 'non_target' for gesture in y_true_str])
            y_pred_macro = np.array([gesture if gesture in TARGET_GESTURES else 'non_target' for gesture in y_pred_str])
            
            macro_f1 = f1_score(y_true_macro, y_pred_macro, average='macro', zero_division='warn')
            
        else:
            # ラベルエンコーダーがない場合は、通常のF1スコアを計算
            if verbose:
                print("Label encoderなし - 数値ラベルで直接計算")
            
            binary_f1 = f1_score(y_true, y_pred, average='binary', zero_division='warn')
            macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division='warn')
        
        # 3. 最終スコア = Binary F1 + Macro F1の平均
        cmi_score = (binary_f1 + macro_f1) / 2.0
        
        if verbose:
            print(f"Binary F1: {binary_f1:.4f}")
            print(f"Macro F1: {macro_f1:.4f}")
            print(f"CMI Score: {cmi_score:.4f}")
            print(f"Test Accuracy: {test_accuracy:.4f}")
        
        return cmi_score, binary_f1, macro_f1, test_accuracy
        
    except Exception as e:
        print(f"CMI評価指標計算でエラー: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0.0, 0.0, 0.0, 0.0


def get_target_gestures():
    """ターゲットジェスチャー一覧を取得"""
    return TARGET_GESTURES.copy()


def get_non_target_gestures():
    """非ターゲットジェスチャー一覧を取得"""
    return NON_TARGET_GESTURES.copy()


def is_target_gesture(gesture_name):
    """指定されたジェスチャーがターゲットかどうかを判定"""
    return gesture_name in TARGET_GESTURES


def print_gesture_info():
    """ジェスチャー情報を出力"""
    print("=== CMI Competition Gesture Information ===")
    print(f"Target Gestures ({len(TARGET_GESTURES)}):")
    for i, gesture in enumerate(TARGET_GESTURES, 1):
        print(f"  {i:2d}. {gesture}")
    
    print(f"\nNon-Target Gestures ({len(NON_TARGET_GESTURES)}):")
    for i, gesture in enumerate(NON_TARGET_GESTURES, 1):
        print(f"  {i:2d}. {gesture}")
    
    print(f"\nTotal: {len(TARGET_GESTURES) + len(NON_TARGET_GESTURES)} gestures")


if __name__ == "__main__":
    # モジュールテスト
    print_gesture_info() 