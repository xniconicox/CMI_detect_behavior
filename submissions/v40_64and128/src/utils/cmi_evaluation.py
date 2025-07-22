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
            # ラベルエンコーダーがない場合は、マルチクラス分類用のF1スコアを計算
            if verbose:
                print("Label encoderなし - マルチクラス分類用F1スコアを計算")
            
            # マルチクラス分類ではbinaryは使用できないため、microまたはmacroを使用
            # ここではmacroを使用（各クラスを平等に扱う）
            binary_f1 = f1_score(y_true, y_pred, average='macro', zero_division='warn')
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


def add_cmi_metrics_to_history(history_dict, y_true, y_pred, label_encoder=None, verbose=False):
    """
    学習履歴辞書にCMI評価指標を追加
    
    Parameters:
    -----------
    history_dict : dict
        学習履歴辞書
    y_true : array-like
        真のラベル
    y_pred : array-like
        予測ラベル
    label_encoder : LabelEncoder, optional
        ラベルエンコーダー
    verbose : bool, default=False
        詳細ログの出力フラグ
    
    Returns:
    --------
    dict
        CMI評価指標が追加された学習履歴辞書
    """
    try:
        # CMI評価指標を計算
        cmi_score, binary_f1, macro_f1, accuracy = calculate_cmi_score(
            y_pred, y_true, label_encoder, verbose
        )
        
        # 新しい形式の学習履歴に対応
        if 'training_metrics' in history_dict and 'validation_metrics' in history_dict:
            # 新しい詳細形式
            if 'cmi_score' not in history_dict['training_metrics']:
                history_dict['training_metrics']['cmi_score'] = []
            if 'binary_f1' not in history_dict['training_metrics']:
                history_dict['training_metrics']['binary_f1'] = []
            if 'macro_f1' not in history_dict['training_metrics']:
                history_dict['training_metrics']['macro_f1'] = []
            
            # 検証データのCMI評価指標も追加（同じ値を使用）
            if 'cmi_score' not in history_dict['validation_metrics']:
                history_dict['validation_metrics']['cmi_score'] = []
            if 'binary_f1' not in history_dict['validation_metrics']:
                history_dict['validation_metrics']['binary_f1'] = []
            if 'macro_f1' not in history_dict['validation_metrics']:
                history_dict['validation_metrics']['macro_f1'] = []
            
            # 各エポックに同じ値を追加（実際の学習では各エポックで異なる値になる）
            epochs = len(history_dict['training_metrics']['loss'])
            for _ in range(epochs):
                history_dict['training_metrics']['cmi_score'].append(cmi_score)
                history_dict['training_metrics']['binary_f1'].append(binary_f1)
                history_dict['training_metrics']['macro_f1'].append(macro_f1)
                history_dict['validation_metrics']['cmi_score'].append(cmi_score)
                history_dict['validation_metrics']['binary_f1'].append(binary_f1)
                history_dict['validation_metrics']['macro_f1'].append(macro_f1)
        
        else:
            # 古い形式
            if 'cmi_score' not in history_dict:
                history_dict['cmi_score'] = []
            if 'binary_f1' not in history_dict:
                history_dict['binary_f1'] = []
            if 'macro_f1' not in history_dict:
                history_dict['macro_f1'] = []
            
            epochs = len(history_dict['loss'])
            for _ in range(epochs):
                history_dict['cmi_score'].append(cmi_score)
                history_dict['binary_f1'].append(binary_f1)
                history_dict['macro_f1'].append(macro_f1)
        
        if verbose:
            print(f"CMI評価指標を学習履歴に追加しました:")
            print(f"  CMI Score: {cmi_score:.4f}")
            print(f"  Binary F1: {binary_f1:.4f}")
            print(f"  Macro F1: {macro_f1:.4f}")
            print(f"  Accuracy: {accuracy:.4f}")
        
        return history_dict
        
    except Exception as e:
        print(f"CMI評価指標の追加でエラー: {str(e)}")
        return history_dict


def create_cmi_metrics_history(y_true, y_pred, label_encoder=None, verbose=False):
    """
    CMI評価指標のみの学習履歴を作成
    
    Parameters:
    -----------
    y_true : array-like
        真のラベル
    y_pred : array-like
        予測ラベル
    label_encoder : LabelEncoder, optional
        ラベルエンコーダー
    verbose : bool, default=False
        詳細ログの出力フラグ
    
    Returns:
    --------
    dict
        CMI評価指標の学習履歴辞書
    """
    try:
        # CMI評価指標を計算
        cmi_score, binary_f1, macro_f1, accuracy = calculate_cmi_score(
            y_pred, y_true, label_encoder, verbose
        )
        
        # 学習履歴辞書を作成
        history_dict = {
            'training_metrics': {
                'cmi_score': [cmi_score],
                'binary_f1': [binary_f1],
                'macro_f1': [macro_f1],
                'accuracy': [accuracy]
            },
            'validation_metrics': {
                'cmi_score': [cmi_score],
                'binary_f1': [binary_f1],
                'macro_f1': [macro_f1],
                'accuracy': [accuracy]
            },
            'metadata': {
                'timestamp': '2025-01-01 00:00:00',
                'model_config': {
                    'num_classes': len(np.unique(y_true)),
                    'label_encoder_used': label_encoder is not None
                }
            },
            'summary': {
                'final_cmi_score': cmi_score,
                'final_binary_f1': binary_f1,
                'final_macro_f1': macro_f1,
                'final_accuracy': accuracy
            }
        }
        
        if verbose:
            print(f"CMI評価指標の学習履歴を作成しました:")
            print(f"  CMI Score: {cmi_score:.4f}")
            print(f"  Binary F1: {binary_f1:.4f}")
            print(f"  Macro F1: {macro_f1:.4f}")
            print(f"  Accuracy: {accuracy:.4f}")
        
        return history_dict
        
    except Exception as e:
        print(f"CMI評価指標の学習履歴作成でエラー: {str(e)}")
        return {}


if __name__ == "__main__":
    # モジュールテスト
    print_gesture_info()
    
    print("\n" + "="*60)
    print("CMI評価指標モジュールの使用方法")
    print("="*60)
    
    print("""
【基本的な使用方法】

1. CMI評価指標の計算:
   from utils.cmi_evaluation import calculate_cmi_score
   
   cmi_score, binary_f1, macro_f1, accuracy = calculate_cmi_score(
       y_pred, y_true, label_encoder, verbose=True
   )

2. ターゲットジェスチャーの取得:
   from utils.cmi_evaluation import get_target_gestures, get_non_target_gestures
   
   target_gestures = get_target_gestures()
   non_target_gestures = get_non_target_gestures()

3. ジェスチャーの判定:
   from utils.cmi_evaluation import is_target_gesture
   
   is_target = is_target_gesture("Above ear - pull hair")

4. 学習履歴にCMI評価指標を追加:
   from utils.cmi_evaluation import add_cmi_metrics_to_history
   
   updated_history = add_cmi_metrics_to_history(
       history_dict, y_true, y_pred, label_encoder, verbose=True
   )

【CMI評価指標の詳細】

- CMI Score: (Binary F1 + Macro F1) / 2.0
- Binary F1: ターゲット vs 非ターゲットの2値分類F1スコア
- Macro F1: 全ジェスチャーのマクロF1スコア（非ターゲットは統合）

【ターゲットジェスチャー（8種類）】
- Above ear - pull hair
- Cheek - pinch skin
- Eyebrow - pull hair
- Eyelash - pull hair
- Forehead - pull hairline
- Forehead - scratch
- Neck - pinch skin
- Neck - scratch

【非ターゲットジェスチャー（10種類）】
- Write name on leg
- Wave hello
- Glasses on/off
- Text on phone
- Write name in air
- Feel around in tray and pull out an object
- Scratch knee/leg skin
- Pull air toward your face
- Drink from bottle/cup
- Pinch knee/leg skin

【関連スクリプト】

学習履歴の可視化:
python src/scripts/visualize_training_history.py --history path/to/training_history.json

CMI評価指標の追加:
python src/scripts/add_cmi_metrics_to_history.py --demo --output demo_history.json
""") 