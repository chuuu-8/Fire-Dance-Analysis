from data_collector import FireDanceDataCollector
import pandas as pd
import os


def quick_label_videos():
    """快速標註訓練影片"""

    # ============ 在這裡修改你的標註 ============
    video_labels = [
        {
            "path": "training_videos/practice1.mp4",
            "segments": [
                {"start": 0, "end": 10, "move": "水平掃動"},
                {"start": 10, "end": 20, "move": "垂直掃動"},
                {"start": 20, "end": 30, "move": "POI旋轉"},
            ]
        },
        # 如果有更多影片，繼續添加：
        # {
        #     "path": "training_videos/practice2.mp4",
        #     "segments": [
        #         {"start": 0, "end": 15, "move": "8字形"},
        #     ]
        # },
    ]
    # ============================================

    # 動作名稱對應
    move_mapping = {
        '水平掃動': 'horizontal_sweep',
        '垂直掃動': 'vertical_sweep',
        'POI旋轉': 'poi_spin',
        '8字形': 'figure8',
        '風車': 'windmill',
        '蝴蝶': 'butterfly',
        '無限符號': 'infinity',
        '分離': 'isolation',
    }

    collector = FireDanceDataCollector()
    all_features = []

    print("開始收集訓練數據...")

    for video_info in video_labels:
        video_path = video_info["path"]

        if not os.path.exists(video_path):
            print(f"❌ 找不到影片: {video_path}")
            continue

        print(f"\n📹 處理影片: {video_path}")

        for seg in video_info["segments"]:
            move_zh = seg['move']
            move_en = move_mapping.get(move_zh, move_zh)

            print(f"  ⏱️  收集 {seg['start']}-{seg['end']}s: {move_zh} ({move_en})")

            features = collector.collect_features_segment(
                video_path,
                os.path.basename(video_path),
                start_sec=seg["start"],
                end_sec=seg["end"],
                sample_every=15,  # 每15幀取一次（約0.5秒）
                max_resolution=480
            )

            # 添加動作標籤
            for f in features:
                f['move'] = move_en

            all_features.extend(features)
            print(f"     ✓ 收集了 {len(features)} 筆特徵")

    # 保存標註數據
    if all_features:
        df = pd.DataFrame(all_features)
        output_file = "training_data/training_dataset.csv"
        df.to_csv(output_file, index=False)

        print(f"\n{'=' * 60}")
        print(f"✅ 成功！已保存 {len(all_features)} 筆訓練數據")
        print(f"📁 檔案位置: {output_file}")
        print(f"\n📊 動作分布:")
        print(df['move'].value_counts())
        print(f"{'=' * 60}")
        print(f"\n🚀 下一步：執行 python train_model.py 來訓練模型")
    else:
        print("\n❌ 沒有收集到任何數據")
        print("請檢查：")
        print("1. 影片是否放在 training_videos/ 資料夾")
        print("2. 影片檔名是否正確")
        print("3. 影片格式是否支援（.mp4, .avi, .mov）")


if __name__ == "__main__":
    quick_label_videos()