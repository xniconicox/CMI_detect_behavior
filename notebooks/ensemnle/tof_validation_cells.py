# ToFの前処理改良分の可視化確認セル

## 6. ToFセンサーデータの詳細分析（改良版）

# ToFセンサーデータの詳細分析
if train_tof_voxel is not None:
    print("=== ToFセンサーデータの詳細分析 ===")
    
    # データ形状の詳細確認
    n_samples, n_depth, n_height, n_width = train_tof_voxel.shape
    print(f"\n【ToFデータ形状】")
    print(f"  サンプル数: {n_samples:,}")
    print(f"  深度方向: {n_depth}")
    print(f"  高さ方向: {n_height}")
    print(f"  幅方向: {n_width}")
    print(f"  総ボクセル数: {n_depth * n_height * n_width}")
    
    # 基本統計
    print(f"\n【基本統計】")
    print(f"  平均値: {train_tof_voxel.mean():.6f}")
    print(f"  標準偏差: {train_tof_voxel.std():.6f}")
    print(f"  最小値: {train_tof_voxel.min():.6f}")
    print(f"  最大値: {train_tof_voxel.max():.6f}")
    
    # データ品質チェック
    nan_count = np.isnan(train_tof_voxel).sum()
    inf_count = np.isinf(train_tof_voxel).sum()
    zero_count = (train_tof_voxel == 0).sum()
    negative_count = (train_tof_voxel < 0).sum()
    
    print(f"\n【データ品質】")
    print(f"  NaN値の数: {nan_count}")
    print(f"  Inf値の数: {inf_count}")
    print(f"  ゼロ値の数: {zero_count} ({zero_count/train_tof_voxel.size*100:.2f}%)")
    print(f"  負の値の数: {negative_count} ({negative_count/train_tof_voxel.size*100:.2f}%)")
    
    # 深度方向の統計
    print(f"\n【深度方向の統計】")
    for depth_idx in range(n_depth):
        depth_data = train_tof_voxel[:, depth_idx, :, :]
        print(f"  深度{depth_idx}: 平均={depth_data.mean():.4f}, 標準偏差={depth_data.std():.4f}")
    
    # ランダムサンプルの可視化
    print("\n=== ランダムToFサンプルの可視化 ===")
    random_idx = np.random.randint(0, n_samples)
    sample_tof = train_tof_voxel[random_idx]  # shape: (5, 8, 8)
    
    # 深度方向の可視化
    fig, axes = plt.subplots(1, n_depth, figsize=(20, 4))
    for depth_idx in range(n_depth):
        im = axes[depth_idx].imshow(sample_tof[depth_idx], cmap='viridis')
        axes[depth_idx].set_title(f'Depth {depth_idx}')
        axes[depth_idx].set_xlabel('Width')
        axes[depth_idx].set_ylabel('Height')
        plt.colorbar(im, ax=axes[depth_idx])
    
    plt.suptitle(f'Random ToF Sample (Index: {random_idx})')
    plt.tight_layout()
    plt.show()
    
    # 深度方向の平均値プロット
    plt.figure(figsize=(10, 6))
    depth_means = train_tof_voxel.mean(axis=(0, 2, 3))
    depth_stds = train_tof_voxel.std(axis=(0, 2, 3))
    
    plt.errorbar(range(n_depth), depth_means, yerr=depth_stds, marker='o')
    plt.xlabel('Depth Index')
    plt.ylabel('Mean Value')
    plt.title('ToF Data: Mean Values by Depth')
    plt.grid(True)
    plt.show()
    
else:
    print("❌ ToFデータが読み込まれていません")

## 7. ToFデータのウィンドウ分割状況確認

# ToFデータのウィンドウ分割状況確認
if train_tof_voxel is not None and train_info is not None:
    print("=== ToFデータのウィンドウ分割状況確認 ===")
    
    # 現在のToFデータの状況
    n_tof_samples = train_tof_voxel.shape[0]
    n_windows = train_windows.shape[0] if train_windows is not None else 0
    
    print(f"\n【現在の状況】")
    print(f"  ToFサンプル数: {n_tof_samples:,}")
    print(f"  ウィンドウ数: {n_windows:,}")
    print(f"  比率: {n_tof_samples/n_windows:.2f} (ToF/ウィンドウ)")
    
    # 情報データからウィンドウの詳細を確認
    if train_info:
        info_df = pd.DataFrame(train_info)
        print(f"\n【ウィンドウ情報】")
        print(f"  ユニークsubject数: {info_df['subject'].nunique()}")
        print(f"  ユニークsequence_id数: {info_df['sequence_id'].nunique()}")
        print(f"  ウィンドウ開始位置の範囲: {info_df['start_idx'].min()} - {info_df['start_idx'].max()}")
        print(f"  ウィンドウ終了位置の範囲: {info_df['end_idx'].min()} - {info_df['end_idx'].max()}")
        
        # サンプルウィンドウの詳細
        sample_window = info_df.iloc[0]
        print(f"\n【サンプルウィンドウ詳細】")
        print(f"  Subject: {sample_window['subject']}")
        print(f"  Sequence ID: {sample_window['sequence_id']}")
        print(f"  開始位置: {sample_window['start_idx']}")
        print(f"  終了位置: {sample_window['end_idx']}")
        print(f"  ウィンドウサイズ: {sample_window['end_idx'] - sample_window['start_idx']}")
    
    # ToFデータの時系列構造確認
    print(f"\n【ToFデータの時系列構造】")
    print(f"  形状: {train_tof_voxel.shape}")
    print(f"  データ型: {train_tof_voxel.dtype}")
    
    # ランダムな時系列サンプルの可視化
    print("\n=== ランダムToF時系列サンプルの可視化 ===")
    random_start = np.random.randint(0, max(1, n_tof_samples - 100))
    sample_series = train_tof_voxel[random_start:random_start+100]
    
    # 深度方向の平均値を時系列でプロット
    series_means = sample_series.mean(axis=(1, 2, 3))
    
    plt.figure(figsize=(15, 5))
    plt.plot(series_means)
    plt.xlabel('Time Step')
    plt.ylabel('Mean ToF Value')
    plt.title(f'ToF Time Series Sample (Start: {random_start})')
    plt.grid(True)
    plt.show()
    
    # 各深度の時系列
    fig, axes = plt.subplots(1, n_depth, figsize=(20, 4))
    for depth_idx in range(n_depth):
        depth_series = sample_series[:, depth_idx, :, :].mean(axis=(1, 2))
        axes[depth_idx].plot(depth_series)
        axes[depth_idx].set_title(f'Depth {depth_idx}')
        axes[depth_idx].set_xlabel('Time Step')
        axes[depth_idx].set_ylabel('Mean Value')
        axes[depth_idx].grid(True)
    
    plt.suptitle(f'ToF Time Series by Depth (Start: {random_start})')
    plt.tight_layout()
    plt.show()
    
else:
    print("❌ ToFデータまたは情報データが読み込まれていません")

## 8. ToFウィンドウ分割の改良確認

# ToFウィンドウ分割の改良確認
print("=== ToFウィンドウ分割の改良確認 ===")

# 現在のToFデータの状況
if train_tof_voxel is not None:
    current_tof_shape = train_tof_voxel.shape
    print(f"\n【現在のToFデータ】")
    print(f"  形状: {current_tof_shape}")
    print(f"  サンプル数: {current_tof_shape[0]:,}")
    
    # ウィンドウ分割が必要な理由
    if train_windows is not None:
        window_count = train_windows.shape[0]
        print(f"\n【ウィンドウ分割の必要性】")
        print(f"  IMUウィンドウ数: {window_count:,}")
        print(f"  ToFサンプル数: {current_tof_shape[0]:,}")
        print(f"  データ数不一致: {'❌ 修正が必要' if current_tof_shape[0] != window_count else '✅ 一致'}")
        
        if current_tof_shape[0] != window_count:
            print(f"\n【改良が必要な理由】")
            print(f"  1. モデル統合時にデータ数が一致しない")
            print(f"  2. アンサンブル学習が困難")
            print(f"  3. 特徴量融合が不可能")
            
            print(f"\n【推奨される改良】")
            print(f"  1. ToFデータをIMUと同じウィンドウ分割で処理")
            print(f"  2. ウィンドウサイズ: 128, ストライド: 64")
            print(f"  3. 各ウィンドウ内のToFデータを平均化または特徴量抽出")
            
            # 改良後の想定形状
            n_depth, n_height, n_width = current_tof_shape[1:]
            expected_shape = (window_count, n_depth, n_height, n_width)
            print(f"\n【改良後の想定形状】")
            print(f"  期待される形状: {expected_shape}")
            print(f"  データ数一致: ✅ 予想")
    
    # 改良実装の確認
    print(f"\n【改良実装の確認】")
    print(f"  1. ToFウィンドウ分割関数の実装: {'✅ 完了' if 'create_tof_windows' in globals() else '❌ 未実装'}")
    print(f"  2. 前処理パイプラインの更新: {'✅ 完了' if 'tof_windows' in locals() else '❌ 未実装'}")
    print(f"  3. データ形状の統一: {'✅ 完了' if train_tof_voxel.shape[0] == train_windows.shape[0] else '❌ 未実装'}")
    
else:
    print("❌ ToFデータが読み込まれていません")

## 9. 改良後のToFデータ品質チェック

# 改良後のToFデータ品質チェック
print("=== 改良後のToFデータ品質チェック ===")

# 改良後のToFデータが存在するかチェック
if 'train_tof_windows' in locals() or 'tof_windows' in locals():
    tof_windows_data = locals().get('train_tof_windows') or locals().get('tof_windows')
    
    if tof_windows_data is not None:
        print(f"\n【改良後のToFデータ】")
        print(f"  形状: {tof_windows_data.shape}")
        print(f"  ウィンドウ数: {tof_windows_data.shape[0]}")
        print(f"  データ型: {tof_windows_data.dtype}")
        
        # 基本統計
        print(f"\n【基本統計】")
        print(f"  平均値: {tof_windows_data.mean():.6f}")
        print(f"  標準偏差: {tof_windows_data.std():.6f}")
        print(f"  最小値: {tof_windows_data.min():.6f}")
        print(f"  最大値: {tof_windows_data.max():.6f}")
        
        # データ品質
        nan_count = np.isnan(tof_windows_data).sum()
        inf_count = np.isinf(tof_windows_data).sum()
        print(f"\n【データ品質】")
        print(f"  NaN値の数: {nan_count}")
        print(f"  Inf値の数: {inf_count}")
        
        # ウィンドウ間の一貫性チェック
        window_means = tof_windows_data.mean(axis=(1, 2, 3))
        window_stds = tof_windows_data.std(axis=(1, 2, 3))
        
        print(f"\n【ウィンドウ間の一貫性】")
        print(f"  平均値の範囲: {window_means.min():.4f} - {window_means.max():.4f}")
        print(f"  標準偏差の範囲: {window_stds.min():.4f} - {window_stds.max():.4f}")
        
        # 改良効果の可視化
        if train_windows is not None:
            print(f"\n【改良効果】")
            print(f"  IMUウィンドウ数: {train_windows.shape[0]}")
            print(f"  ToFウィンドウ数: {tof_windows_data.shape[0]}")
            print(f"  データ数一致: {'✅ 成功' if train_windows.shape[0] == tof_windows_data.shape[0] else '❌ 失敗'}")
            
            # ランダムウィンドウの可視化
            random_idx = np.random.randint(0, tof_windows_data.shape[0])
            sample_tof_window = tof_windows_data[random_idx]
            
            fig, axes = plt.subplots(1, sample_tof_window.shape[0], figsize=(20, 4))
            for depth_idx in range(sample_tof_window.shape[0]):
                im = axes[depth_idx].imshow(sample_tof_window[depth_idx], cmap='viridis')
                axes[depth_idx].set_title(f'Depth {depth_idx}')
                plt.colorbar(im, ax=axes[depth_idx])
            
            plt.suptitle(f'Improved ToF Window (Index: {random_idx})')
            plt.tight_layout()
            plt.show()
    
    else:
        print("❌ 改良後のToFデータが見つかりません")
        print("  前処理パイプラインの改良が必要です")

else:
    print("❌ 改良後のToFデータが読み込まれていません")
    print("  前処理パイプラインの改良が必要です") 