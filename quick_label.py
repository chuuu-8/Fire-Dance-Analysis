import cv2
import mediapipe as mp
import csv
import os
import numpy as np
import json
import joblib

# 載入招式定義
try:
    with open('move_config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
        MOVES_MAP = config.get('annotation_keys', {})
        MOVES_NAME_MAP = config.get('moves', {})
except Exception as e:
    print(f"[ERROR] 無法載入設定檔: {e}")
    MOVES_MAP = {}
    MOVES_NAME_MAP = {}


class DataLabeler:
    def __init__(self, output_file='training_data/training_dataset.csv'):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.output_file = output_file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # 檢查是否需要寫入標頭
        if not os.path.exists(output_file):
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['move', 'features'])

        # 載入 FireDanceAnalyzer
        try:
            from fire_dance_analyzer import FireDanceAnalyzer
            self.analyzer = FireDanceAnalyzer()
            self.analyzer.reset_history()
        except ImportError:
            print("[ERROR] 找不到 fire_dance_analyzer.py！")
            self.analyzer = None

        self.recorded_count = 0

    def extract_features(self, landmarks):
        if self.analyzer:
            return self.analyzer.extract_pose_features(landmarks)
        else:
            return None

    def _save_data(self, label, features):
        feature_str = json.dumps(features.tolist())
        with open(self.output_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([label, feature_str])

    def process_manual(self, video_path):
        """手動模式：按住按鍵才錄製"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] 無法開啟影片: {video_path}")
            return

        print(f"[INFO] 手動模式: {video_path}")
        print("  - 按住招式鍵錄製，放開停止")

        if self.analyzer: self.analyzer.reset_history()

        current_label = None
        is_recording = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.resize(frame, (1280, 720))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)

            key = cv2.waitKey(10) & 0xFF
            if key == 27 or key == ord('q'): break

            char_key = chr(key).upper() if 0 <= key <= 127 else ''

            if char_key in MOVES_MAP:
                current_label = MOVES_MAP[char_key]
                is_recording = True
            else:
                is_recording = False
                current_label = None

            if results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                features = self.extract_features(results.pose_landmarks)

                if is_recording and features is not None and current_label:
                    self._save_data(current_label, features)
                    cv2.putText(frame, f"RECORDING: {current_label}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2)
                    self.recorded_count += 1

            cv2.imshow('Manual Labeler', frame)

        cap.release()
        cv2.destroyAllWindows()

    def process_auto(self, video_path, target_key):
        """自動模式：整部影片都標記為同一招式"""
        if target_key not in MOVES_MAP:
            print(f"[ERROR] 無效的按鍵代號: {target_key}")
            return

        target_label = MOVES_MAP[target_key]
        move_name = MOVES_NAME_MAP.get(target_label, target_label)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] 無法開啟影片: {video_path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[INFO] 自動模式: {video_path}")
        print(f"[INFO] 目標招式: {move_name} ({target_label})")
        print(f"[INFO] 總幀數: {total_frames}，處理中請稍候...")

        if self.analyzer: self.analyzer.reset_history()

        local_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # 雖然是自動，但還是要跑姿勢偵測來抓特徵
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)

            if results.pose_landmarks:
                features = self.extract_features(results.pose_landmarks)
                if features is not None:
                    self._save_data(target_label, features)
                    local_count += 1
                    self.recorded_count += 1

            if local_count % 50 == 0:
                print(f"  已處理: {local_count}/{total_frames} 幀...", end='\r')

        cap.release()
        print(f"\n[INFO] {move_name} 處理完成！新增 {local_count} 筆數據。")


if __name__ == "__main__":
    print("=" * 40)
    print("火舞數據標註工具 (Quick Labeler)")
    print("=" * 40)

    # 顯示招式列表供參考
    print("可用招式代碼:")
    for k, v in MOVES_MAP.items():
        if len(k) == 1:
            name = MOVES_NAME_MAP.get(v, v)
            print(f"  {k}: {name}")
    print("-" * 40)

    print("請選擇模式:")
    print("1. 手動模式 (觀看影片並按住按鍵錄製)")
    print("2. 自動模式 (指定一部影片為特定招式，整部自動轉換)")

    mode = input("請輸入模式 (1 或 2): ").strip()

    if mode == '2':
        while True:
            print("\n" + "-" * 30)
            video_path = input("請輸入影片路徑 (輸入 q 結束): ").strip()
            if video_path.lower() == 'q': break

            if not os.path.exists(video_path):
                print("[ERROR] 找不到檔案，請重新輸入")
                continue

            key = input("請輸入這部影片代表的招式按鍵 (例如 1, 2, A...): ").strip().upper()
            if key not in MOVES_MAP:
                print("[ERROR] 代碼錯誤，請參考上方列表")
                continue

            labeler = DataLabeler()
            labeler.process_auto(video_path, key)
            print(f"[INFO] 目前總累積數據量: {labeler.recorded_count}")

    else:
        video_path = input("請輸入影片路徑: ").strip()
        if os.path.exists(video_path):
            labeler = DataLabeler()
            labeler.process_manual(video_path)
        else:
            print("[ERROR] 找不到檔案")
