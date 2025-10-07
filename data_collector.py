import cv2
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
from fire_dance_analyzer import FireDanceAnalyzer

class FireDanceDataCollector:
    def __init__(self, output_dir="training_data"):
        self.analyzer = FireDanceAnalyzer()
        self.output_dir = output_dir
        self.data_file = os.path.join(output_dir, "pose_features.csv")
        self.annotations_file = os.path.join(output_dir, "annotations.json")
        
        # 確保輸出目錄存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化數據存儲
        self.features_data = []
        self.annotations = {}
        
        # 載入現有數據
        self.load_existing_data()
    
    def load_existing_data(self):
        """載入現有的數據"""
        if os.path.exists(self.data_file):
            self.features_data = pd.read_csv(self.data_file).to_dict('records')
            print(f"已載入 {len(self.features_data)} 條現有數據")
        
        if os.path.exists(self.annotations_file):
            with open(self.annotations_file, 'r', encoding='utf-8') as f:
                self.annotations = json.load(f)
            print(f"已載入 {len(self.annotations)} 條標註數據")
    
    def collect_features_from_video(self, video_path, video_name, sample_every=24, max_resolution=240):
        """從影片中收集姿勢特徵
        sample_every: 每 N 幀取 1 幀分析（預設 24）
        max_resolution: 影片高度超過時等比縮到此高度（預設 240p）
        """
        print(f"正在分析影片: {video_name}")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 計算縮放尺寸
        if orig_h and orig_h > max_resolution:
            scale = max_resolution / orig_h
            new_w = int(orig_w * scale)
            new_h = max_resolution
        else:
            new_w, new_h = orig_w, orig_h
        
        frame_count = 0
        collected_features = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 依採樣間隔取樣
            if frame_count % sample_every != 0:
                continue
            
            # 降解析度（如需要）
            if new_w and new_h and (new_w != orig_w or new_h != orig_h):
                frame = cv2.resize(frame, (new_w, new_h))
            
            # 轉換顏色空間
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.analyzer.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                features = self.analyzer.extract_pose_features(results.pose_landmarks)
                if features is not None:
                    time_sec = frame_count / (fps if fps else 30.0)
                    collected_features.append({
                        'video_name': video_name,
                        'frame': frame_count,
                        'time': round(time_sec, 2),
                        'features': features.tolist()
                    })
            
            # 顯示進度
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"進度: {progress:.1f}%")
        
        cap.release()
        print(f"從 {video_name} 收集了 {len(collected_features)} 條特徵數據")
        return collected_features

    def collect_features_segment(self, video_path, video_name, start_sec, end_sec, sample_every=8, max_resolution=480):
        """只在指定時間片段收集姿勢特徵，並可調整採樣率與最大解析度以加速。
        start_sec/end_sec: 以秒為單位。
        sample_every: 每 N 幀取 1 幀分析。
        max_resolution: 高於此高度時會等比例縮到此高度。
        """
        print(f"片段分析: {video_name} [{start_sec:.2f}s ~ {end_sec:.2f}s], 每{sample_every}幀取1, <= {max_resolution}p")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 定位到起始時間
        cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)

        collected = []
        frame_idx = int(start_sec * fps)

        # 計算縮放
        scale = 1.0
        if orig_h > max_resolution and orig_h > 0:
            scale = max_resolution / orig_h
            new_w = int(orig_w * scale)
            new_h = max_resolution
        else:
            new_w, new_h = orig_w, orig_h

        while cap.isOpened():
            cur_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            cur_sec = cur_ms / 1000.0
            if cur_sec >= end_sec:
                break
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % sample_every != 0:
                continue
            if new_w != orig_w or new_h != orig_h:
                frame = cv2.resize(frame, (new_w, new_h))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.analyzer.pose.process(rgb)
            if results.pose_landmarks:
                features = self.analyzer.extract_pose_features(results.pose_landmarks)
                if features is not None:
                    collected.append({
                        'video_name': video_name,
                        'frame': frame_idx,
                        'time': round(cur_sec, 2),
                        'features': features.tolist()
                    })
        cap.release()
        print(f"片段收集完成，共 {len(collected)} 筆")
        return collected
    
    def manual_annotation_tool(self, video_path, video_name):
        """手動標註工具"""
        print(f"開始手動標註影片: {video_name}")
        print("使用鍵盤控制:")
        print("數字鍵 1-0: 選擇招式")
        print("空格鍵: 暫停/播放")
        print("ESC: 退出")
        print("S: 保存當前標註")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 招式映射
        move_keys = {
            ord('1'): 'spin',
            ord('2'): 'throw', 
            ord('3'): 'figure8',
            ord('4'): 'circle',
            ord('5'): 'wave',
            ord('6'): 'cross',
            ord('7'): 'behind_back',
            ord('8'): 'under_leg',
            ord('9'): 'overhead',
            ord('0'): 'side_arm'
        }
        
        current_move = None
        annotations = []
        paused = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
            
            # 顯示當前招式
            if current_move:
                move_text = f"當前招式: {self.analyzer.get_move_description(current_move)}"
                cv2.putText(frame, move_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 顯示控制說明
            cv2.putText(frame, "1-0: 選擇招式, 空格: 暫停, ESC: 退出", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Fire Dance Annotation Tool', frame)
            
            key = cv2.waitKey(30) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord(' '):  # 空格
                paused = not paused
            elif key in move_keys:
                current_move = move_keys[key]
                current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                annotations.append({
                    'time': round(current_time, 2),
                    'move': current_move
                })
                print(f"標註: {current_time:.2f}s - {self.analyzer.get_move_description(current_move)}")
            elif key == ord('s'):  # 保存
                self.save_annotations(video_name, annotations)
                print("標註已保存")
        
        cap.release()
        cv2.destroyAllWindows()
        
        return annotations
    
    def save_annotations(self, video_name, annotations):
        """保存標註數據"""
        self.annotations[video_name] = annotations
        
        with open(self.annotations_file, 'w', encoding='utf-8') as f:
            json.dump(self.annotations, f, ensure_ascii=False, indent=2)
    
    def create_training_dataset(self):
        """創建訓練數據集"""
        print("正在創建訓練數據集...")
        
        training_data = []
        
        for video_name, annotations in self.annotations.items():
            # 找到對應的特徵數據
            video_features = [f for f in self.features_data if f['video_name'] == video_name]
            
            for feature_record in video_features:
                # 找到最接近時間點的標註
                closest_annotation = None
                min_time_diff = float('inf')
                
                for annotation in annotations:
                    time_diff = abs(feature_record['time'] - annotation['time'])
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        closest_annotation = annotation
                
                # 如果時間差小於1秒，使用該標註
                if closest_annotation and min_time_diff < 1.0:
                    training_record = {
                        'video_name': video_name,
                        'frame': feature_record['frame'],
                        'time': feature_record['time'],
                        'features': feature_record['features'],
                        'move': closest_annotation['move']
                    }
                    training_data.append(training_record)
        
        # 保存訓練數據
        training_df = pd.DataFrame(training_data)
        training_file = os.path.join(self.output_dir, "training_dataset.csv")
        training_df.to_csv(training_file, index=False)
        
        print(f"訓練數據集已保存: {training_file}")
        print(f"總共 {len(training_data)} 條訓練數據")
        
        # 顯示數據分布
        if training_data:
            move_counts = training_df['move'].value_counts()
            print("\n招式數據分布:")
            for move, count in move_counts.items():
                print(f"{self.analyzer.get_move_description(move)}: {count}")
        
        return training_data
    
    def batch_process_videos(self, video_dir):
        """批量處理影片目錄"""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        
        for filename in os.listdir(video_dir):
            if any(filename.lower().endswith(ext) for ext in video_extensions):
                video_path = os.path.join(video_dir, filename)
                video_name = os.path.splitext(filename)[0]
                
                print(f"\n處理影片: {filename}")
                
                # 收集特徵
                features = self.collect_features_from_video(video_path, video_name)
                self.features_data.extend(features)
                
                # 詢問是否要標註
                response = input(f"是否要標註影片 {filename}? (y/n): ")
                if response.lower() == 'y':
                    annotations = self.manual_annotation_tool(video_path, video_name)
                    self.save_annotations(video_name, annotations)
        
        # 保存所有特徵數據
        features_df = pd.DataFrame(self.features_data)
        features_df.to_csv(self.data_file, index=False)
        print(f"\n所有特徵數據已保存到: {self.data_file}")

    def collect_directory_noninteractive(self, video_dirs):
        """非互動式批次處理一個或多個影片目錄，僅收集姿勢特徵並保存。
        參數 video_dirs 可為字串或字串列表。
        """
        if isinstance(video_dirs, str):
            video_dirs = [video_dirs]
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        total_videos = 0
        total_records = 0
        for video_dir in video_dirs:
            if not os.path.exists(video_dir):
                print(f"目錄不存在，略過: {video_dir}")
                continue
            print(f"掃描目錄: {video_dir}")
            for filename in os.listdir(video_dir):
                if any(filename.lower().endswith(ext) for ext in video_extensions):
                    video_path = os.path.join(video_dir, filename)
                    video_name = os.path.splitext(filename)[0]
                    print(f"\n處理影片: {filename}")
                    features = self.collect_features_from_video(video_path, video_name)
                    self.features_data.extend(features)
                    total_videos += 1
                    total_records += len(features)
        # 保存所有特徵數據
        if self.features_data:
            features_df = pd.DataFrame(self.features_data)
            features_df.to_csv(self.data_file, index=False)
            print(f"\n所有特徵數據已保存到: {self.data_file}")
            print(f"影片數: {total_videos}, 特徵筆數: {total_records}")
        else:
            print("未收集到任何特徵數據。")

def run_noninteractive(dirs):
    """從外部方便呼叫的非互動式入口。"""
    collector = FireDanceDataCollector()
    collector.collect_directory_noninteractive(dirs)

def main():
    """主函數"""
    collector = FireDanceDataCollector()
    
    print("火舞數據收集工具")
    print("1. 批量處理影片目錄")
    print("2. 手動標註單個影片")
    print("3. 創建訓練數據集")
    print("4. 退出")
    
    while True:
        choice = input("\n請選擇操作 (1-4): ")
        
        if choice == '1':
            video_dir = input("請輸入影片目錄路徑: ")
            if os.path.exists(video_dir):
                collector.batch_process_videos(video_dir)
            else:
                print("目錄不存在")
        
        elif choice == '2':
            video_path = input("請輸入影片路徑: ")
            if os.path.exists(video_path):
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                collector.manual_annotation_tool(video_path, video_name)
            else:
                print("檔案不存在")
        
        elif choice == '3':
            collector.create_training_dataset()
        
        elif choice == '4':
            break
        
        else:
            print("無效選擇")

if __name__ == "__main__":
    main()

def smooth_trajectory(trajectory, window=5):
    """軌跡平滑化"""
    trajectory = list(trajectory)  # 保證切片安全
    if len(trajectory) < window:
        return trajectory
    smoothed = []
    for i in range(len(trajectory)):
        start = max(0, i - window // 2)
        end = min(len(trajectory), i + window // 2 + 1)
        avg_point = np.mean(trajectory[start:end], axis=0)
        smoothed.append(tuple(map(int, avg_point)))
    return smoothed
