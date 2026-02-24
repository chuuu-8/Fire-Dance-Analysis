#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import mediapipe as mp
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from fire_dance_analyzer import FireDanceAnalyzer

# 設定 Matplotlib 支援中文顯示 (避免亂碼)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


class FireDanceModelTrainer:
    def __init__(self, data_dir="training_data", video_dir="training_videos"):
        self.data_dir = data_dir
        self.video_dir = video_dir
        self.analyzer = FireDanceAnalyzer()
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_score = 0

        # 確保資料目錄存在
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def extract_features_from_videos(self):
        """從影片資料夾提取特徵並儲存為 CSV"""
        print(f"正在掃描影片目錄: {self.video_dir} ...")

        if not os.path.exists(self.video_dir):
            print(f"錯誤: 找不到影片目錄 {self.video_dir}")
            return False

        # 定義資料結構
        data_records = []

        # 準備 Mediapipe
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 遍歷所有子資料夾 (招式名稱)
        for move_name in os.listdir(self.video_dir):
            move_path = os.path.join(self.video_dir, move_name)
            if not os.path.isdir(move_path):
                continue

            print(f"📂 正在處理招式類別: {move_name}...")

            # 遍歷影片
            for filename in os.listdir(move_path):
                if not filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    continue

                video_path = os.path.join(move_path, filename)
                cap = cv2.VideoCapture(video_path)

                # 重置分析器歷史 (這很重要，因為每個影片是獨立的)
                self.analyzer.reset_history()

                frame_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break

                    # 加速處理：每 3 幀取樣一次
                    if frame_count % 3 != 0:
                        frame_count += 1
                        continue

                    # 影像前處理
                    h, w = frame.shape[:2]
                    # 如果影片太大，縮小一點加快速度
                    if w > 640:
                        scale = 640 / w
                        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(rgb)

                    if results.pose_landmarks:
                        # 提取特徵
                        features = self.analyzer.extract_pose_features(results.pose_landmarks)

                        if features is not None and not np.isnan(features).any():
                            # 將 numpy array 轉為 list 存入 CSV
                            data_records.append({
                                'move': move_name,
                                'features': features.tolist(),  # 存成 list 方便後續讀取
                                'source_video': filename
                            })

                    frame_count += 1
                cap.release()

        pose.close()

        if not data_records:
            print("❌ 沒有提取到任何數據！請檢查影片是否包含人物。")
            return False

        # 轉成 DataFrame 並存檔
        df = pd.DataFrame(data_records)
        csv_path = os.path.join(self.data_dir, "training_dataset.csv")
        df.to_csv(csv_path, index=False)
        print(f"✅ 數據提取完成！共 {len(df)} 筆數據，已儲存至 {csv_path}")
        return True

    def load_training_data(self):
        """載入訓練數據 (如果 CSV 不存在或需要更新，則先從影片提取)"""
        csv_path = os.path.join(self.data_dir, "training_dataset.csv")

        # 檢查是否需要重新從影片提取數據
        # 如果 CSV 不存在，或者 training_videos 資料夾存在，我們就預設重新掃描一次影片以確保數據最新
        if not os.path.exists(csv_path) or os.path.exists(self.video_dir):
            print("檢測到影片資料夾，開始更新訓練數據...")
            success = self.extract_features_from_videos()
            if not success and not os.path.exists(csv_path):
                return None

        if not os.path.exists(csv_path):
            print(f"訓練數據文件不存在: {csv_path}")
            return None

        # 載入數據
        df = pd.read_csv(csv_path)
        print(f"\n載入 CSV 數據: {len(df)} 條")

        # 檢查數據分布
        print("招式數據分布:")
        move_counts = df['move'].value_counts()
        for move, count in move_counts.items():
            print(f"  {move} ({self.analyzer.get_move_description(move)}): {count}")

        return df

    def prepare_features(self, df):
        """準備特徵數據"""
        # CSV 中的 list 讀出來可能是字串，需要轉換
        features = []
        for feature_val in df['features']:
            if isinstance(feature_val, str):
                # 如果是字串格式 '[0.1, 0.2]'
                feature_str = feature_val.strip('[]')
                # 處理可能的換行符或多餘空格
                feature_array = [float(x) for x in feature_str.replace('\n', '').split(',')]
            else:
                # 如果已經是 list (某些 pandas 版本讀取行為)
                feature_array = feature_val
            features.append(feature_array)

        X = np.array(features)
        y = df['move'].values

        print(f"\n數據準備完成:")
        print(f"  特徵維度: {X.shape}")
        print(f"  標籤數量: {len(y)}")

        return X, y

    def train_multiple_models(self, X, y):
        """訓練多個模型並比較性能"""
        # 分割數據
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 標準化特徵
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 定義模型
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, max_depth=6, random_state=42
            ),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(100, 50), max_iter=500, random_state=42
            )
        }

        results = {}

        print("\n開始訓練並比較模型...")
        print("=" * 60)

        for name, model in models.items():
            print(f"正在訓練 {name}...")

            # 訓練模型
            model.fit(X_train_scaled, y_train)

            # 評估模型
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)

            # 交叉驗證 (CV)
            try:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
            except:
                cv_mean = 0
                cv_std = 0

            results[name] = {
                'model': model,
                'train_score': train_score,
                'test_score': test_score,
                'cv_mean': cv_mean,
                'cv_std': cv_std
            }

            print(f"  -> 測試準確率: {test_score:.3f} | CV: {cv_mean:.3f}")

        return results, X_test_scaled, y_test

    def select_best_model(self, results):
        """選擇最佳模型"""
        print("\n模型性能總結:")
        print("-" * 60)
        print(f"{'模型名稱':<20} | {'測試準確率':<10} | {'交叉驗證(CV)':<10}")
        print("-" * 60)

        best_model_name = None
        best_score = 0

        for name, result in results.items():
            score = result['test_score']
            print(f"{name:<20} | {score:.3f}      | {result['cv_mean']:.3f}")

            # 這裡以測試分數為主，若分數相同可比較 CV
            if score > best_score:
                best_score = score
                best_model_name = name

        print("-" * 60)
        print(f"🏆 最佳模型: {best_model_name} (準確率: {best_score:.3f})")

        self.best_model = results[best_model_name]['model']
        self.best_score = best_score

        return best_model_name, results[best_model_name]

    def evaluate_model(self, model, X_test, y_test):
        """詳細評估模型 (繪製混淆矩陣)"""
        y_pred = model.predict(X_test)

        print("\n詳細分類報告:")
        print("-" * 50)

        # 取得所有出現過的標籤
        unique_labels = sorted(list(set(y_test) | set(y_pred)))

        # 嘗試取得中文名稱
        target_names = [self.analyzer.get_move_description(lbl) for lbl in unique_labels]

        print(classification_report(y_test, y_pred, target_names=target_names))

        # 繪製混淆矩陣
        try:
            cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=target_names,
                        yticklabels=target_names)
            plt.title('混淆矩陣 (Confusion Matrix)')
            plt.xlabel('預測結果')
            plt.ylabel('真實標籤')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()

            # 儲存圖片
            save_path = os.path.join(self.data_dir, 'confusion_matrix.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 混淆矩陣圖表已儲存至: {save_path}")
            # plt.show() # 如果是在伺服器跑，建議註解掉這行
            plt.close()
        except Exception as e:
            print(f"繪圖時發生錯誤 (可能不影響模型儲存): {e}")

        return y_pred

    def save_model(self, model_name="fire_dance_model.pkl"):
        """保存模型"""
        if self.best_model is None:
            print("沒有可保存的模型")
            return

        # 取得目前 analyzer 的招式表 (從影片資料夾名稱來的)
        # 我們需要確保 moves 字典包含所有訓練過的類別
        current_moves = self.analyzer.moves

        model_data = {
            'classifier': self.best_model,
            'scaler': self.scaler,
            'moves': current_moves,  # 這邊會儲存招式對照表
            'training_info': {
                'best_score': self.best_score,
                'date': str(pd.Timestamp.now())
            }
        }

        joblib.dump(model_data, model_name)
        print(f"\n💾 模型已成功保存到: {model_name}")
        print("現在您可以重新啟動 app.py 來使用新模型了！")

    def run_training(self):
        """執行完整的訓練流程"""
        # 載入數據 (包含自動從影片提取)
        df = self.load_training_data()
        if df is None:
            print("無法載入數據，訓練終止。")
            return

        # 準備特徵
        X, y = self.prepare_features(df)

        # 訓練多個模型
        results, X_test, y_test = self.train_multiple_models(X, y)

        # 選擇最佳模型
        best_model_name, best_result = self.select_best_model(results)

        # 詳細評估
        self.evaluate_model(best_result['model'], X_test, y_test)

        # 保存模型
        self.save_model()


def main():
    # 確保這裡的目錄名稱對應您的資料夾
    trainer = FireDanceModelTrainer(
        data_dir="training_data",  # 用來存 CSV 和圖片的地方
        video_dir="training_videos"  # 您的影片資料夾
    )
    trainer.run_training()


if __name__ == "__main__":
    main()
