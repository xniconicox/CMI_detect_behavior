"""
Demographics特徴量エンジニアリング最適化モジュール

LSTM v2の性能向上のためのdemographics特徴量の最適化を行います。
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DemographicsOptimizer:
    """
    Demographics特徴量エンジニアリング最適化クラス
    """
    
    def __init__(self, random_state=42):
        """
        初期化
        
        Parameters:
        -----------
        random_state : int
            乱数シード
        """
        self.random_state = random_state
        self.feature_transformers = {}
        self.feature_selectors = {}
        self.optimization_results = {}
        
    def load_demographics_data(self, data_path):
        """
        Demographics データ読み込み
        
        Parameters:
        -----------
        data_path : str
            データディレクトリパス
        
        Returns:
        --------
        dict : 読み込まれたデータ
        """
        print("Demographics データ読み込み中...")
        
        # LSTM v2の前処理済みデータから読み込み
        import pickle
        from sklearn.model_selection import train_test_split
        
        # データファイルパス
        demo_path = os.path.join(data_path, 'X_demographics_windows.pkl')
        label_path = os.path.join(data_path, 'y_windows.pkl')
        
        # データ読み込み
        with open(demo_path, 'rb') as f:
            X_demographics_all = pickle.load(f)
        
        with open(label_path, 'rb') as f:
            y_all = pickle.load(f)
        
        # numpy配列に変換
        X_demographics_all = np.array(X_demographics_all, dtype=np.float32)
        y_all = np.array(y_all, dtype=np.int32)
        
        # 訓練・検証データに分割（8:2）
        X_demo_train, X_demo_val, y_train, y_val = train_test_split(
            X_demographics_all, y_all, 
            test_size=0.2, 
            random_state=self.random_state,
            stratify=y_all
        )
        
        print(f"訓練データ: {X_demo_train.shape}")
        print(f"検証データ: {X_demo_val.shape}")
        print(f"ラベル: {len(np.unique(y_train))}クラス")
        
        return {
            'X_demo_train': X_demo_train,
            'X_demo_val': X_demo_val,
            'y_train': y_train,
            'y_val': y_val
        }
    
    def analyze_feature_importance(self, X_demo, y):
        """
        特徴量重要度分析
        
        Parameters:
        -----------
        X_demo : np.ndarray
            Demographics特徴量
        y : np.ndarray
            ラベル
        
        Returns:
        --------
        dict : 特徴量重要度結果
        """
        print("特徴量重要度分析中...")
        
        results = {}
        
        # 1. F統計量による重要度
        f_selector = SelectKBest(score_func=f_classif, k='all')
        f_selector.fit(X_demo, y)
        f_scores = f_selector.scores_
        f_pvalues = f_selector.pvalues_
        
        # 2. 相互情報量による重要度
        mi_scores = mutual_info_classif(X_demo, y, random_state=self.random_state)
        
        # 3. RandomForestによる重要度
        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        rf.fit(X_demo, y)
        rf_importance = rf.feature_importances_
        
        # 結果統合
        n_features = X_demo.shape[1]
        feature_importance = pd.DataFrame({
            'feature_idx': range(n_features),
            'f_score': f_scores,
            'f_pvalue': f_pvalues,
            'mi_score': mi_scores,
            'rf_importance': rf_importance
        })
        
        # 正規化スコア
        feature_importance['f_score_norm'] = (f_scores - f_scores.min()) / (f_scores.max() - f_scores.min())
        feature_importance['mi_score_norm'] = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min())
        feature_importance['rf_importance_norm'] = rf_importance / rf_importance.max()
        
        # 総合スコア（平均）
        feature_importance['combined_score'] = (
            feature_importance['f_score_norm'] + 
            feature_importance['mi_score_norm'] + 
            feature_importance['rf_importance_norm']
        ) / 3
        
        # 重要度順にソート
        feature_importance = feature_importance.sort_values('combined_score', ascending=False)
        
        results['feature_importance'] = feature_importance
        results['top_features'] = feature_importance.head(10)['feature_idx'].tolist()
        
        print(f"最重要特徴量上位10: {results['top_features']}")
        
        return results
    
    def optimize_feature_selection(self, X_demo_train, y_train, X_demo_val, y_val):
        """
        特徴量選択最適化
        
        Parameters:
        -----------
        X_demo_train : np.ndarray
            訓練用demographics特徴量
        y_train : np.ndarray
            訓練用ラベル
        X_demo_val : np.ndarray
            検証用demographics特徴量
        y_val : np.ndarray
            検証用ラベル
        
        Returns:
        --------
        dict : 最適化結果
        """
        print("特徴量選択最適化中...")
        
        results = {}
        
        # ベースライン性能（全特徴量）
        rf_baseline = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        baseline_scores = cross_val_score(rf_baseline, X_demo_train, y_train, cv=5, scoring='f1_macro')
        baseline_mean = baseline_scores.mean()
        baseline_std = baseline_scores.std()
        
        print(f"ベースライン性能: {baseline_mean:.4f} ± {baseline_std:.4f}")
        
        # 特徴量数を変えて最適化
        n_features_range = [5, 8, 10, 12, 15, 18, 20]  # 元の20特徴量から選択
        
        best_score = 0
        best_k = 0
        best_selector = None
        
        for k in n_features_range:
            if k >= X_demo_train.shape[1]:
                continue
                
            # SelectKBestで特徴量選択
            selector = SelectKBest(score_func=f_classif, k=k)
            X_train_selected = selector.fit_transform(X_demo_train, y_train)
            
            # 性能評価
            rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            scores = cross_val_score(rf, X_train_selected, y_train, cv=5, scoring='f1_macro')
            mean_score = scores.mean()
            std_score = scores.std()
            
            print(f"k={k}: {mean_score:.4f} ± {std_score:.4f}")
            
            if mean_score > best_score:
                best_score = mean_score
                best_k = k
                best_selector = selector
        
        # 最適な特徴量選択器を保存
        self.feature_selectors['best_k_selector'] = best_selector
        
        results['baseline_score'] = baseline_mean
        results['best_k'] = best_k
        results['best_score'] = best_score
        results['improvement'] = best_score - baseline_mean
        results['selected_features'] = best_selector.get_support(indices=True).tolist()
        
        print(f"最適特徴量数: {best_k}")
        print(f"最適性能: {best_score:.4f}")
        print(f"改善度: {results['improvement']:+.4f}")
        
        return results
    
    def optimize_feature_engineering(self, X_demo_train, y_train):
        """
        特徴量エンジニアリング最適化
        
        Parameters:
        -----------
        X_demo_train : np.ndarray
            訓練用demographics特徴量
        y_train : np.ndarray
            訓練用ラベル
        
        Returns:
        --------
        dict : 最適化結果
        """
        print("特徴量エンジニアリング最適化中...")
        
        results = {}
        
        # 1. 多項式特徴量生成
        print("多項式特徴量生成テスト...")
        poly_results = {}
        
        for degree in [2]:  # 2次のみテスト（計算量考慮）
            poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=True)
            X_poly = poly.fit_transform(X_demo_train)
            
            # 特徴量数が多すぎる場合は制限
            if X_poly.shape[1] > 100:
                # 重要な特徴量のみ選択
                selector = SelectKBest(score_func=f_classif, k=50)
                X_poly = selector.fit_transform(X_poly, y_train)
                poly_selector = selector
            else:
                poly_selector = None
            
            # 性能評価
            rf = RandomForestClassifier(n_estimators=50, random_state=self.random_state)
            scores = cross_val_score(rf, X_poly, y_train, cv=3, scoring='f1_macro')
            mean_score = scores.mean()
            
            poly_results[degree] = {
                'score': mean_score,
                'n_features': X_poly.shape[1],
                'transformer': poly,
                'selector': poly_selector
            }
            
            print(f"Degree {degree}: {mean_score:.4f} ({X_poly.shape[1]} features)")
        
        # 2. PCA次元削減
        print("PCA次元削減テスト...")
        pca_results = {}
        
        for n_components in [10, 15, 18]:
            if n_components >= X_demo_train.shape[1]:
                continue
                
            pca = PCA(n_components=n_components, random_state=self.random_state)
            X_pca = pca.fit_transform(X_demo_train)
            
            # 性能評価
            rf = RandomForestClassifier(n_estimators=50, random_state=self.random_state)
            scores = cross_val_score(rf, X_pca, y_train, cv=3, scoring='f1_macro')
            mean_score = scores.mean()
            
            pca_results[n_components] = {
                'score': mean_score,
                'explained_variance_ratio': pca.explained_variance_ratio_.sum(),
                'transformer': pca
            }
            
            print(f"PCA {n_components}: {mean_score:.4f} (累積寄与率: {pca.explained_variance_ratio_.sum():.3f})")
        
        results['polynomial_features'] = poly_results
        results['pca_results'] = pca_results
        
        # 最良の手法を選択
        best_method = 'original'
        best_score = 0
        
        # ベースライン
        rf_baseline = RandomForestClassifier(n_estimators=50, random_state=self.random_state)
        baseline_scores = cross_val_score(rf_baseline, X_demo_train, y_train, cv=3, scoring='f1_macro')
        baseline_score = baseline_scores.mean()
        best_score = baseline_score
        
        # 多項式特徴量の最良結果
        if poly_results:
            best_poly = max(poly_results.items(), key=lambda x: x[1]['score'])
            if best_poly[1]['score'] > best_score:
                best_score = best_poly[1]['score']
                best_method = f'polynomial_degree_{best_poly[0]}'
        
        # PCAの最良結果
        if pca_results:
            best_pca = max(pca_results.items(), key=lambda x: x[1]['score'])
            if best_pca[1]['score'] > best_score:
                best_score = best_pca[1]['score']
                best_method = f'pca_{best_pca[0]}'
        
        results['baseline_score'] = baseline_score
        results['best_method'] = best_method
        results['best_score'] = best_score
        results['improvement'] = best_score - baseline_score
        
        print(f"最良手法: {best_method}")
        print(f"最良性能: {best_score:.4f}")
        print(f"改善度: {results['improvement']:+.4f}")
        
        return results
    
    def create_optimized_features(self, X_demo_train, X_demo_val, y_train, 
                                 feature_selection_results, feature_engineering_results):
        """
        最適化された特徴量を作成
        
        Parameters:
        -----------
        X_demo_train : np.ndarray
            訓練用demographics特徴量
        X_demo_val : np.ndarray
            検証用demographics特徴量
        y_train : np.ndarray
            訓練用ラベル
        feature_selection_results : dict
            特徴量選択結果
        feature_engineering_results : dict
            特徴量エンジニアリング結果
        
        Returns:
        --------
        dict : 最適化された特徴量
        """
        print("最適化された特徴量を作成中...")
        
        # 特徴量選択を適用
        if 'best_k_selector' in self.feature_selectors:
            selector = self.feature_selectors['best_k_selector']
            X_train_selected = selector.transform(X_demo_train)
            X_val_selected = selector.transform(X_demo_val)
        else:
            X_train_selected = X_demo_train
            X_val_selected = X_demo_val
        
        # 特徴量エンジニアリングを適用
        best_method = feature_engineering_results['best_method']
        
        if best_method.startswith('polynomial'):
            degree = int(best_method.split('_')[-1])
            poly_results = feature_engineering_results['polynomial_features'][degree]
            
            # 多項式特徴量生成
            poly = poly_results['transformer']
            X_train_poly = poly.transform(X_train_selected)
            X_val_poly = poly.transform(X_val_selected)
            
            # 特徴量選択（必要に応じて）
            if poly_results['selector'] is not None:
                selector = poly_results['selector']
                X_train_final = selector.transform(X_train_poly)
                X_val_final = selector.transform(X_val_poly)
            else:
                X_train_final = X_train_poly
                X_val_final = X_val_poly
                
        elif best_method.startswith('pca'):
            n_components = int(best_method.split('_')[-1])
            pca_results = feature_engineering_results['pca_results'][n_components]
            
            # PCA変換（特徴量選択前の元データを使用）
            pca = pca_results['transformer']
            X_train_final = pca.transform(X_demo_train)
            X_val_final = pca.transform(X_demo_val)
            
        else:
            # オリジナル特徴量
            X_train_final = X_train_selected
            X_val_final = X_val_selected
        
        print(f"最適化後の特徴量形状: 訓練{X_train_final.shape}, 検証{X_val_final.shape}")
        
        return {
            'X_demo_train_optimized': X_train_final,
            'X_demo_val_optimized': X_val_final,
            'optimization_method': best_method,
            'original_shape': X_demo_train.shape,
            'optimized_shape': X_train_final.shape
        }
    
    def run_optimization(self, data_path, output_path):
        """
        Demographics特徴量最適化を実行
        
        Parameters:
        -----------
        data_path : str
            データディレクトリパス
        output_path : str
            出力ディレクトリパス
        
        Returns:
        --------
        dict : 最適化結果
        """
        print("Demographics特徴量最適化を開始...")
        print(f"データパス: {data_path}")
        print(f"出力パス: {output_path}")
        
        # 出力ディレクトリ作成
        os.makedirs(output_path, exist_ok=True)
        
        # データ読み込み
        data = self.load_demographics_data(data_path)
        
        # 特徴量重要度分析
        importance_results = self.analyze_feature_importance(
            data['X_demo_train'], data['y_train']
        )
        
        # 特徴量選択最適化
        selection_results = self.optimize_feature_selection(
            data['X_demo_train'], data['y_train'],
            data['X_demo_val'], data['y_val']
        )
        
        # 特徴量エンジニアリング最適化
        engineering_results = self.optimize_feature_engineering(
            data['X_demo_train'], data['y_train']
        )
        
        # 最適化された特徴量作成
        optimized_features = self.create_optimized_features(
            data['X_demo_train'], data['X_demo_val'], data['y_train'],
            selection_results, engineering_results
        )
        
        # 結果統合
        results = {
            'timestamp': datetime.now().isoformat(),
            'data_info': {
                'train_shape': data['X_demo_train'].shape,
                'val_shape': data['X_demo_val'].shape,
                'n_classes': len(np.unique(data['y_train']))
            },
            'feature_importance': {
                'top_features': importance_results['top_features'],
                'feature_scores': importance_results['feature_importance'].to_dict('records')
            },
            'feature_selection': selection_results,
            'feature_engineering': {
                'best_method': engineering_results['best_method'],
                'best_score': engineering_results['best_score'],
                'improvement': engineering_results['improvement']
            },
            'optimized_features': {
                'method': optimized_features['optimization_method'],
                'original_shape': optimized_features['original_shape'],
                'optimized_shape': optimized_features['optimized_shape']
            }
        }
        
        # 結果保存
        results_file = os.path.join(output_path, 'demographics_optimization_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 最適化された特徴量保存
        np.save(os.path.join(output_path, 'X_demographics_train_optimized.npy'), 
                optimized_features['X_demo_train_optimized'])
        np.save(os.path.join(output_path, 'X_demographics_val_optimized.npy'), 
                optimized_features['X_demo_val_optimized'])
        
        print("Demographics特徴量最適化完了")
        print(f"結果保存: {results_file}")
        
        return results

def main():
    """
    メイン実行関数
    """
    # パス設定
    data_path = '../output/experiments/lstm_v2_w64_s16/preprocessed'
    output_path = '../results/demographics_optimization'
    
    # 最適化実行
    optimizer = DemographicsOptimizer(random_state=42)
    results = optimizer.run_optimization(data_path, output_path)
    
    # 結果表示
    print("\n" + "="*50)
    print("Demographics特徴量最適化結果")
    print("="*50)
    print(f"最良特徴量エンジニアリング: {results['feature_engineering']['best_method']}")
    print(f"性能改善: {results['feature_engineering']['improvement']:+.4f}")
    print(f"最適特徴量数: {results['feature_selection']['best_k']}")
    print(f"最適化後形状: {results['optimized_features']['optimized_shape']}")
    print("="*50)

if __name__ == "__main__":
    main() 