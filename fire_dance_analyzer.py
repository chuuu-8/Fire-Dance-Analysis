import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os
from collections import deque

class FireDanceAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # 降低模型複雜度 (0=最快, 1=平衡, 2=最準確)
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 火舞招式定義
        self.moves = {
            'spin': '旋轉',
            'throw': '拋接',
            'figure8': '八字形',
            'circle': '圓形',
            'wave': '波浪',
            'cross': '交叉',
            'behind_back': '背後',
            'under_leg': '腿下',
            'overhead': '頭上',
            'side_arm': '側臂',
            'stall': '停球',
            'pendulum': '鐘擺'
        }
        
        self.classifier = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def extract_pose_features(self, landmarks):
        """提取姿勢特徵"""
        if not landmarks:
            return None
            
        features = []
        
        # 獲取關鍵點座標
        points = []
        for landmark in landmarks.landmark:
            points.append([landmark.x, landmark.y, landmark.visibility])
        
        points = np.array(points)
        
        # 1. 手臂角度特徵
        # 左臂角度 (肩膀-手肘-手腕)
        left_arm_angle = self._calculate_angle(points[11], points[13], points[15])
        # 右臂角度
        right_arm_angle = self._calculate_angle(points[12], points[14], points[16])
        
        # 2. 身體姿態特徵
        # 身體傾斜角度
        body_tilt = self._calculate_body_tilt(points[11], points[12], points[23], points[24])
        
        # 3. 手臂相對位置
        # 左臂相對於身體的位置
        left_arm_position = self._calculate_arm_position(points[11], points[13], points[15])
        # 右臂相對於身體的位置
        right_arm_position = self._calculate_arm_position(points[12], points[14], points[16])
        
        # 4. 手部運動軌跡特徵
        # 手部高度
        left_hand_height = points[15][1]
        right_hand_height = points[16][1]
        
        # 5. 身體旋轉特徵
        body_rotation = self._calculate_body_rotation(points)
        
        # 6. 手臂伸展程度
        left_arm_extension = self._calculate_arm_extension(points[11], points[13], points[15])
        right_arm_extension = self._calculate_arm_extension(points[12], points[14], points[16])
        
        # 7. 手部交叉檢測
        hands_crossed = self._detect_hands_crossed(points[15], points[16])
        
        # 8. 手臂在身體後方檢測
        left_arm_behind = self._detect_arm_behind_body(points[11], points[13], points[15], points[23], points[24])
        right_arm_behind = self._detect_arm_behind_body(points[12], points[14], points[16], points[23], points[24])
        
        # 9. 手臂在頭上檢測
        left_arm_overhead = self._detect_arm_overhead(points[15], points[11])
        right_arm_overhead = self._detect_arm_overhead(points[16], points[12])
        
        # 10. 手臂在腿下檢測
        left_arm_under_leg = self._detect_arm_under_leg(points[15], points[23], points[25])
        right_arm_under_leg = self._detect_arm_under_leg(points[16], points[24], points[26])
        
        features = [
            left_arm_angle, right_arm_angle,
            body_tilt,
            left_arm_position, right_arm_position,
            left_hand_height, right_hand_height,
            body_rotation,
            left_arm_extension, right_arm_extension,
            hands_crossed,
            left_arm_behind, right_arm_behind,
            left_arm_overhead, right_arm_overhead,
            left_arm_under_leg, right_arm_under_leg
        ]
        
        return np.array(features)
    
    def _calculate_angle(self, point1, point2, point3):
        """計算三點之間的角度"""
        a = np.array([point1[0], point1[1]])
        b = np.array([point2[0], point2[1]])
        c = np.array([point3[0], point3[1]])
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def _calculate_body_tilt(self, left_shoulder, right_shoulder, left_hip, right_hip):
        """計算身體傾斜角度"""
        shoulder_center = (left_shoulder[:2] + right_shoulder[:2]) / 2
        hip_center = (left_hip[:2] + right_hip[:2]) / 2
        
        dx = shoulder_center[0] - hip_center[0]
        dy = shoulder_center[1] - hip_center[1]
        
        return np.arctan2(dx, dy)
    
    def _calculate_arm_position(self, shoulder, elbow, wrist):
        """計算手臂相對於身體的位置"""
        # 簡化為手臂相對於肩膀的水平位置
        return wrist[0] - shoulder[0]
    
    def _calculate_body_rotation(self, points):
        """計算身體旋轉角度"""
        # 基於肩膀和髖部的相對位置
        shoulder_center = (points[11][:2] + points[12][:2]) / 2
        hip_center = (points[23][:2] + points[24][:2]) / 2
        
        dx = shoulder_center[0] - hip_center[0]
        return dx
    
    def _calculate_arm_extension(self, shoulder, elbow, wrist):
        """計算手臂伸展程度"""
        # 計算手臂的總長度
        arm_length = np.linalg.norm(shoulder[:2] - elbow[:2]) + np.linalg.norm(elbow[:2] - wrist[:2])
        return arm_length
    
    def _detect_hands_crossed(self, left_hand, right_hand):
        """檢測手部是否交叉"""
        # 簡化檢測：比較手部的水平位置
        return 1 if left_hand[0] > right_hand[0] else 0
    
    def _detect_arm_behind_body(self, shoulder, elbow, wrist, left_hip, right_hip):
        """檢測手臂是否在身體後方"""
        # 簡化檢測：比較手腕和髖部的水平位置
        hip_center_x = (left_hip[0] + right_hip[0]) / 2
        return 1 if abs(wrist[0] - hip_center_x) > 0.1 else 0
    
    def _detect_arm_overhead(self, hand, shoulder):
        """檢測手臂是否在頭上"""
        return 1 if hand[1] < shoulder[1] else 0
    
    def _detect_arm_under_leg(self, hand, hip, knee):
        """檢測手臂是否在腿下"""
        # 簡化檢測：比較手部和膝蓋的垂直位置
        return 1 if hand[1] > knee[1] else 0
    
    def analyze_frame_sequence(self, video_path, sample_rate=5, max_resolution=720):
        """分析影片幀序列並識別招式"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 降低解析度以提升速度
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if original_height > max_resolution:
            scale_factor = max_resolution / original_height
            new_width = int(original_width * scale_factor)
            new_height = max_resolution
            print(f"降低解析度: {original_width}x{original_height} -> {new_width}x{new_height}")
        else:
            new_width, new_height = original_width, original_height
        
        frame_count = 0
        analyzed_frames = 0
        move_timeline = []
        features_sequence = []
        
        # 用於未訓練情況下的啟發式偵測狀態
        rotation_history = deque(maxlen=15)
        current_candidate_move = None
        stable_count = 0
        last_committed_move = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # 每 sample_rate 幀分析一次
            if frame_count % sample_rate != 0:
                continue
                
            # 降低解析度
            if new_width != original_width:
                frame = cv2.resize(frame, (new_width, new_height))
                
            # 轉換顏色空間
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # 提取特徵
                features = self.extract_pose_features(results.pose_landmarks)
                if features is not None:
                    features_sequence.append(features)
                    
                    # 如果模型已訓練，進行招式預測
                    if self.is_trained:
                        # 標準化特徵
                        features_scaled = self.scaler.transform(features.reshape(1, -1))
                        prediction = self.classifier.predict(features_scaled)[0]
                        confidence = np.max(self.classifier.predict_proba(features_scaled))
                        
                        # 只保留高置信度的預測
                        if confidence > 0.6:
                            time_sec = frame_count / fps
                            move_timeline.append({
                                "time": round(time_sec, 2),
                                "move": prediction,
                                "confidence": round(confidence, 3)
                            })
                    else:
                        # 未訓練：使用啟發式規則做基礎偵測
                        heuristic_move, heuristic_conf, rotation_history = self._heuristic_predict(features, rotation_history)
                        if heuristic_move is not None:
                            # 穩定判斷：需要連續多幀一致才記錄
                            if heuristic_move == current_candidate_move:
                                stable_count += 1
                            else:
                                current_candidate_move = heuristic_move
                                stable_count = 1
                            
                            if stable_count >= 5 and current_candidate_move != last_committed_move:
                                time_sec = frame_count / fps
                                move_timeline.append({
                                    "time": round(time_sec, 2),
                                    "move": current_candidate_move,
                                    "confidence": round(float(heuristic_conf), 3)
                                })
                                last_committed_move = current_candidate_move
            
            analyzed_frames += 1
            
        cap.release()
        
        return move_timeline, features_sequence

    def _heuristic_predict(self, features: np.ndarray, rotation_history: deque):
        """根據幾何特徵進行簡單啟發式招式判斷。
        回傳: (move_name or None, confidence, updated_rotation_history)
        """
        # 特徵索引對應
        left_arm_angle = features[0]
        right_arm_angle = features[1]
        body_tilt = features[2]
        left_pos = features[3]
        right_pos = features[4]
        left_hand_y = features[5]
        right_hand_y = features[6]
        body_rot = features[7]
        left_ext = features[8]
        right_ext = features[9]
        hands_crossed = features[10]
        left_behind = features[11]
        right_behind = features[12]
        left_over = features[13]
        right_over = features[14]
        left_under = features[15]
        right_under = features[16]

        # 更新旋轉歷史
        rotation_history.append(body_rot)
        move = None
        conf = 0.0

        # 1) 頭上 (任一手在頭上且伸展較大)
        if (left_over == 1 and left_ext > 0.35) or (right_over == 1 and right_ext > 0.35):
            move = 'overhead'
            conf = 0.8

        # 2) 腿下 (任一手在腿下)
        elif left_under == 1 or right_under == 1:
            move = 'under_leg'
            conf = 0.8

        # 3) 交叉
        elif hands_crossed == 1 and (left_ext > 0.3 or right_ext > 0.3):
            move = 'cross'
            conf = 0.7

        # 4) 背後 (任一手在身體後方)
        elif left_behind == 1 or right_behind == 1:
            move = 'behind_back'
            conf = 0.7

        # 5) 側臂 (一側手臂伸直且與身體呈開展)
        elif (left_ext > 0.38 and abs(left_pos) > 0.12) or (right_ext > 0.38 and abs(right_pos) > 0.12):
            move = 'side_arm'
            conf = 0.65

        # 6) 旋轉 (身體旋轉值在短時間內有明顯振盪)
        else:
            if len(rotation_history) >= 8:
                rot_arr = np.array(rotation_history)
                amplitude = rot_arr.max() - rot_arr.min()
                # 設定一個相對寬鬆的閾值 (在 mediapipe 正規化座標下)
                if amplitude > 0.18:
                    move = 'spin'
                    conf = min(0.6 + amplitude, 0.85)

        return move, conf, rotation_history
    
    def train_model(self, training_data_path):
        """訓練招式識別模型"""
        # 這裡需要準備訓練數據
        # 格式：features, label
        # 暫時使用模擬數據進行演示
        
        print("開始訓練火舞招式識別模型...")
        
        # 創建模擬訓練數據
        n_samples = 1000
        n_features = 17
        
        # 生成模擬特徵數據
        X = np.random.randn(n_samples, n_features)
        
        # 生成模擬標籤
        move_names = list(self.moves.keys())
        y = np.random.choice(move_names, n_samples)
        
        # 分割訓練和測試數據
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 標準化特徵
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 訓練隨機森林分類器
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self.classifier.fit(X_train_scaled, y_train)
        
        # 評估模型
        train_score = self.classifier.score(X_train_scaled, y_train)
        test_score = self.classifier.score(X_test_scaled, y_test)
        
        print(f"訓練準確率: {train_score:.3f}")
        print(f"測試準確率: {test_score:.3f}")
        
        self.is_trained = True
        
        # 保存模型
        self.save_model('fire_dance_model.pkl')
        
        return train_score, test_score
    
    def save_model(self, filepath):
        """保存模型"""
        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'moves': self.moves
        }
        joblib.dump(model_data, filepath)
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath):
        """載入模型"""
        if os.path.exists(filepath):
            model_data = joblib.load(filepath)
            self.classifier = model_data['classifier']
            self.scaler = model_data['scaler']
            self.moves = model_data['moves']
            self.is_trained = True
            print(f"模型已從 {filepath} 載入")
            return True
        else:
            print(f"模型檔案 {filepath} 不存在")
            return False
    
    def get_move_description(self, move_name):
        """獲取招式的中文描述"""
        return self.moves.get(move_name, move_name)
