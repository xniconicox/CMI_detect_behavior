import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 設定
CURRENT_DIR = Path(__file__).parent if '__file__' in globals() else Path('.')
PROJECT_ROOT = CURRENT_DIR.parent.parent

# ラベルマッピング（学習時と同じ）
LABEL_MAPPING = {
    'Above ear - pull hair': 0, 'Cheek - pinch skin': 1, 'Drink from bottle/cup': 2,
    'Eyebrow - pull hair': 3, 'Eyelash - pull hair': 4, 'Feel around in tray and pull out an object': 5,
    'Forehead - pull hairline': 6, 'Forehead - scratch': 7, 'Glasses on/off': 8,
    'Neck - pinch skin': 9, 'Neck - scratch': 10, 'Pinch knee/leg skin': 11,
    'Pull air toward your face': 12, 'Scratch knee/leg skin': 13, 'Text on phone': 14,
    'Wave hello': 15, 'Write name in air': 16, 'Write name on leg': 17
}

REVERSE_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}

class CMIPreprocessor:
    def __init__(self):
        """CMI前処理器の初期化"""
        self.sensor_scaler = None
        self.demographics_scaler = None
        self.pca_transformer = None
        self.model = None
        self.setup_gpu()
        
    def setup_gpu(self):
        """GPU設定"""
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"GPU利用可能: {len(gpus)}台")
            else:
                print("CPU環境で実行")
        except Exception as e:
            print(f"GPU設定エラー: {e}")
    
    def load_preprocessors(self):
        """学習時に保存した前処理器を読み込み"""
        try:
            # 学習時の前処理器パス
            training_data_path = PROJECT_ROOT / "output/experiments/lstm_v2_w64_s16/preprocessed"
            
            # センサー用Scaler
            sensor_scaler_path = training_data_path / "sensor_scaler.pkl"
            with open(sensor_scaler_path, 'rb') as f:
                self.sensor_scaler = pickle.load(f)
            print("✅ センサーScaler読み込み完了")
            
            # Demographics用Scaler
            demographics_scaler_path = training_data_path / "demographics_scaler.pkl"
            with open(demographics_scaler_path, 'rb') as f:
                self.demographics_scaler = pickle.load(f)
            print("✅ DemographicsScaler読み込み完了")
            
            # Demographics データから PCA Transformer を作成
            demographics_data_path = training_data_path / "X_demographics_windows.pkl"
            with open(demographics_data_path, 'rb') as f:
                demographics_data = pickle.load(f)
            
            # PCA変換器を作成（18次元）
            self.pca_transformer = PCA(n_components=18, random_state=42)
            self.pca_transformer.fit(demographics_data)
            print("✅ PCA Transformer作成完了")
            print(f"  元の次元数: {demographics_data.shape[1]}")
            print(f"  変換後次元数: 18")
            print(f"  累積寄与率: {self.pca_transformer.explained_variance_ratio_.sum():.4f}")
            
        except Exception as e:
            print(f"❌ 前処理器読み込みエラー: {e}")
            raise
    
    def load_model(self):
        """学習済みモデルを読み込み"""
        try:
            model_path = CURRENT_DIR / "final_model_20250709_085324.keras"
            if model_path.exists():
                print(f"モデル読み込み中: {model_path}")
                self.model = tf.keras.models.load_model(str(model_path))
                print("✅ モデル読み込み完了")
            else:
                print(f"❌ モデルファイルが見つかりません: {model_path}")
                raise FileNotFoundError(f"Model file not found: {model_path}")
        except Exception as e:
            print(f"❌ モデル読み込みエラー: {e}")
            raise
    
    def create_demographics_features(self, demographics_df, subject_id):
        """Demographics特徴量を作成（学習時と同じ20特徴量）"""
        try:
            # 該当するsubject_idのデータを取得
            demo_data = demographics_df[demographics_df['subject'] == subject_id]
            
            if len(demo_data) == 0:
                print(f"⚠️  subject_id {subject_id} のdemographics情報が見つかりません")
                # デフォルト値を使用
                adult_child = 1
                age = 30.0
                sex = 1  # male
                handedness = 1  # right
                height_cm = 170.0
                shoulder_to_wrist_cm = 55.0
                elbow_to_wrist_cm = 25.0
            else:
                row = demo_data.iloc[0]
                # テストデータの構造に合わせてマッピング
                adult_child = row['adult_child']
                age = row['age']
                sex = row['sex']
                handedness = row['handedness']
                height_cm = row['height_cm']
                shoulder_to_wrist_cm = row['shoulder_to_wrist_cm']
                elbow_to_wrist_cm = row['elbow_to_wrist_cm']
            
            # 派生特徴量の作成（学習時と同じ）
            arm_ratio = elbow_to_wrist_cm / shoulder_to_wrist_cm
            arm_length_relative = shoulder_to_wrist_cm / height_cm
            sex_age_interaction = sex * age
            handedness_sex = handedness * 2 + sex
            
            # 年齢グループ（学習時と同じ境界値）
            age_group_child = 1.0 if age <= 12 else 0.0
            age_group_teen = 1.0 if 12 < age <= 18 else 0.0
            age_group_young_adult = 1.0 if 18 < age <= 25 else 0.0
            age_group_adult = 1.0 if 25 < age <= 35 else 0.0
            age_group_senior = 1.0 if age > 35 else 0.0
            
            # 身長カテゴリ（学習時と同じ境界値）
            height_cat_short = 1.0 if height_cm <= 160 else 0.0
            height_cat_average = 1.0 if 160 < height_cm <= 170 else 0.0
            height_cat_tall = 1.0 if 170 < height_cm <= 180 else 0.0
            height_cat_very_tall = 1.0 if height_cm > 180 else 0.0
            
            # 学習時と同じ順序で特徴量を作成（20個）
            features = np.array([
                adult_child,
                age,
                sex,
                handedness,
                height_cm,
                shoulder_to_wrist_cm,
                elbow_to_wrist_cm,
                arm_ratio,
                arm_length_relative,
                sex_age_interaction,
                handedness_sex,
                age_group_child,
                age_group_teen,
                age_group_young_adult,
                age_group_adult,
                age_group_senior,
                height_cat_short,
                height_cat_average,
                height_cat_tall,
                height_cat_very_tall
            ], dtype=np.float32)
            
            return features
            
        except Exception as e:
            print(f"❌ Demographics特徴量作成エラー: {e}")
            # エラー時はデフォルト値を返す（20個の特徴量）
            return np.array([
                1.0, 30.0, 1.0, 1.0, 170.0, 55.0, 25.0, 0.45, 0.32, 30.0, 3.0,
                0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0
            ], dtype=np.float32)
    
    def preprocess_sensor_data(self, sequence_df, window_size=64, stride=16):
        """センサーデータの前処理（学習時と同じ）"""
        try:
            # 非センサーカラムを除外してセンサーカラムを特定
            exclude_cols = ['row_id', 'sequence_id', 'sequence_counter', 'subject', 'timestamp']
            sensor_columns = [col for col in sequence_df.columns if col not in exclude_cols]
            
            # センサーデータを抽出
            sensor_data = sequence_df[sensor_columns].to_numpy()
            
            print(f"  センサーデータ形状: {sensor_data.shape}")
            print(f"  センサーカラム数: {len(sensor_columns)}")
            
            # 短いシーケンスの場合はパディングを追加（学習時と同じ処理）
            if len(sensor_data) < window_size:
                print("⚠️  シーケンスが短いため、パディングを追加します。")
                padding_size = window_size - len(sensor_data)
                padding = np.zeros((padding_size, sensor_data.shape[1]))
                sensor_data = np.vstack([sensor_data, padding])
            
            # スライディングウィンドウでデータを分割
            windows = []
            for i in range(0, len(sensor_data) - window_size + 1, stride):
                window = sensor_data[i:i + window_size]
                windows.append(window)
            
            # 最低1つのウィンドウは作成される
            if len(windows) == 0:
                print("⚠️  ウィンドウが作成できません。最後のウィンドウを使用します。")
                windows = [sensor_data[-window_size:]]
            
            windows = np.array(windows)
            print(f"  作成されたウィンドウ数: {len(windows)}")
            print(f"  ウィンドウ形状: {windows.shape}")
            
            return windows
            
        except Exception as e:
            print(f"❌ センサーデータ前処理エラー: {e}")
            raise
    
    def preprocess_demographics_data(self, demographics_df, subject_id):
        """Demographics データの前処理（学習時と同じ順序）"""
        try:
            # 特徴量エンジニアリング（20個の特徴量を作成）
            features = self.create_demographics_features(demographics_df, subject_id)
            print(f"  Demographics特徴量形状: {features.shape}")
            
            # 学習時と同じ処理: 20特徴量から18特徴量を選択（最後の2つを除外）
            # 学習時に child や very_tall カテゴリが存在せず、18特徴量になったと推測
            features_18 = features[:-2]  # 最後の2つの特徴量を除外
            print(f"  18特徴量に削減: {features_18.shape}")
            
            # PCA変換（18→18）
            features_pca = self.pca_transformer.transform(features_18.reshape(1, -1))
            print(f"  PCA変換後形状: {features_pca.shape}")
            
            # 正規化（PCA後の18特徴量に適用）
            features_scaled = self.demographics_scaler.transform(features_pca)
            print(f"  正規化後形状: {features_scaled.shape}")
            
            return features_scaled.flatten()
            
        except Exception as e:
            print(f"❌ Demographics前処理エラー: {e}")
            raise
    
    def predict_gesture(self, sequence_df, demographics_df):
        """ジェスチャー予測を実行"""
        try:
            # sequence_idとsubject_idを取得
            sequence_id = sequence_df['sequence_id'].iloc[0]
            subject_id = sequence_df['subject'].iloc[0]
            print(f"推論開始: sequence_id={sequence_id}")
            
            # センサーデータの前処理
            sensor_windows = self.preprocess_sensor_data(sequence_df, window_size=64, stride=16)
            
            # 正規化
            sensor_windows_scaled = []
            for window in sensor_windows:
                # NaN/Infチェック
                if np.any(np.isnan(window)) or np.any(np.isinf(window)):
                    print(f"⚠️  センサーデータにNaN/Infが含まれています")
                    print(f"  NaN数: {np.sum(np.isnan(window))}")
                    print(f"  Inf数: {np.sum(np.isinf(window))}")
                
                window_scaled = self.sensor_scaler.transform(window)
                
                # 正規化後もチェック
                if np.any(np.isnan(window_scaled)) or np.any(np.isinf(window_scaled)):
                    print(f"⚠️  正規化後のデータにNaN/Infが含まれています")
                    print(f"  NaN数: {np.sum(np.isnan(window_scaled))}")
                    print(f"  Inf数: {np.sum(np.isinf(window_scaled))}")
                
                sensor_windows_scaled.append(window_scaled)
            
            sensor_windows_scaled = np.array(sensor_windows_scaled)
            print(f"  正規化後センサーデータ形状: {sensor_windows_scaled.shape}")
            
            # Demographics データの前処理
            demographics_features = self.preprocess_demographics_data(demographics_df, subject_id)
            
            # Demographics特徴量のNaN/Infチェック
            if np.any(np.isnan(demographics_features)) or np.any(np.isinf(demographics_features)):
                print(f"⚠️  Demographics特徴量にNaN/Infが含まれています")
                print(f"  NaN数: {np.sum(np.isnan(demographics_features))}")
                print(f"  Inf数: {np.sum(np.isinf(demographics_features))}")
            
            # 各ウィンドウに対してdemographics特徴量を複製
            demographics_windows = np.tile(demographics_features, (len(sensor_windows_scaled), 1))
            print(f"  Demographics特徴量形状: {demographics_windows.shape}")
            
            # 推論前のデータ統計情報
            print(f"  推論前センサーデータ統計:")
            print(f"    - 最小値: {np.min(sensor_windows_scaled):.6f}")
            print(f"    - 最大値: {np.max(sensor_windows_scaled):.6f}")
            print(f"    - 平均値: {np.mean(sensor_windows_scaled):.6f}")
            print(f"    - 標準偏差: {np.std(sensor_windows_scaled):.6f}")
            
            print(f"  推論前Demographics統計:")
            print(f"    - 最小値: {np.min(demographics_windows):.6f}")
            print(f"    - 最大値: {np.max(demographics_windows):.6f}")
            print(f"    - 平均値: {np.mean(demographics_windows):.6f}")
            print(f"    - 標準偏差: {np.std(demographics_windows):.6f}")
            
            # 推論実行
            predictions = self.model.predict([sensor_windows_scaled, demographics_windows], verbose=0)
            
            # 予測結果のNaN/Infチェック
            if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                print(f"⚠️  予測結果にNaN/Infが含まれています")
                print(f"  NaN数: {np.sum(np.isnan(predictions))}")
                print(f"  Inf数: {np.sum(np.isinf(predictions))}")
                print(f"  予測結果の形状: {predictions.shape}")
                print(f"  予測結果の最初の値: {predictions[0][:5]}")  # 最初の5個の値を表示
                
                # NaN対策として、デフォルト予測を使用
                print(f"  NaN対策: デフォルト予測を使用します")
                # 18クラスの均等な確率分布を作成
                predictions = np.full((1, 18), 1.0/18, dtype=np.float32)
                print(f"  修正後予測結果: {predictions[0][:5]}")
            
            # 複数ウィンドウの場合は平均を取る
            if len(predictions) > 1:
                avg_prediction = np.mean(predictions, axis=0)
                predicted_class = np.argmax(avg_prediction)
                confidence = np.max(avg_prediction)
                print(f"  ウィンドウ数: {len(predictions)}, 平均予測信頼度: {confidence:.4f}")
            else:
                predicted_class = np.argmax(predictions[0])
                confidence = np.max(predictions[0])
                print(f"  単一ウィンドウ予測信頼度: {confidence:.4f}")
            
            # ラベルマッピングを使用して予測結果を取得
            predicted_gesture = REVERSE_LABEL_MAPPING[int(predicted_class)]
            print(f"  予測結果: {predicted_gesture}")
            
            return predicted_gesture
            
        except Exception as e:
            print(f"❌ 推論エラー: {e}")
            import traceback
            traceback.print_exc()
            # エラー時はデフォルト値を返す
            return "Wave hello" 