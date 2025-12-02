import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os
import json
from collections import deque

# === 修正後的招式定義 ===
ALLOWED_MOVES_DEFAULT = {
    'forward_weave': '正八',
    'backward_weave': '反八',
    'butterfly': '蝴蝶',
    'head_roll': '繞頭',
    'three_beat_weave': '三轉',
    '2beat': '2beat',
    '3beat': '3beat',
    '4beat': '4beat',
    'flower_3petal': '三葉花',
    'flower_4petal': '四葉花',
    'continuous_toss': '連拋',
    'crosser': 'crosser',
    '4petal': '四葉',
    '4petal_iso': '四葉(加iso)',
    'side_4petal': '側四葉',
    'cap': 'cap',
    'stall': '停球',
    'isolation': 'isolation'
}
ALLOWED_MOVE_KEYS = set(ALLOWED_MOVES_DEFAULT.keys())


class FireDanceAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 載入招式定義
        self.moves = self._load_moves_definition()

        self.classifier = None
        self.scaler = StandardScaler()
        self.is_trained = False

        # 用於計算速度的上一幀記錄
        self.prev_landmarks_array = None

    def reset_history(self):
        """重置歷史記錄（在處理新影片開始時調用）"""
        self.prev_landmarks_array = None

    def _load_moves_definition(self):
        """載入招式定義"""
        # 嘗試從模型載入
        model_file = 'fire_dance_model.pkl'
        if os.path.exists(model_file):
            try:
                model_data = joblib.load(model_file)
                if 'moves' in model_data:
                    return {k: v for k, v in model_data['moves'].items() if k in ALLOWED_MOVE_KEYS}
            except Exception:
                pass

        # 嘗試從配置檔載入
        config_file = 'move_config.json'
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    moves = config.get('moves', {})
                    filtered = {k: v for k, v in moves.items() if k in ALLOWED_MOVE_KEYS}
                    if filtered:
                        return filtered
            except Exception:
                pass

        return ALLOWED_MOVES_DEFAULT.copy()

    def extract_pose_features(self, landmarks):
        """提取姿勢特徵（含動態速度特徵）"""
        if not landmarks:
            return None

        # 獲取關鍵點座標
        points = []
        for landmark in landmarks.landmark:
            points.append([landmark.x, landmark.y, landmark.visibility])
        points = np.array(points)

        # === 1. 靜態幾何特徵 ===
        left_arm_angle = self._calculate_angle(points[11], points[13], points[15])
        right_arm_angle = self._calculate_angle(points[12], points[14], points[16])

        left_arm_extension = self._calculate_arm_extension(points[11], points[13], points[15])
        right_arm_extension = self._calculate_arm_extension(points[12], points[14], points[16])

        body_center_x = (points[11][0] + points[12][0]) / 2
        body_center_y = (points[11][1] + points[12][1]) / 2

        left_hand_rel_x = points[15][0] - body_center_x
        left_hand_rel_y = points[15][1] - body_center_y
        right_hand_rel_x = points[16][0] - body_center_x
        right_hand_rel_y = points[16][1] - body_center_y

        left_hand_height = points[15][1]
        right_hand_height = points[16][1]
        hands_height_diff = abs(left_hand_height - right_hand_height)
        hands_distance = np.linalg.norm(points[15][:2] - points[16][:2])

        arm_symmetry = abs(left_arm_angle - right_arm_angle) / 180.0
        body_tilt = self._calculate_body_tilt(points[11], points[12], points[23], points[24])
        shoulder_width = np.linalg.norm(points[11][:2] - points[12][:2])
        hands_crossed = 1 if left_hand_rel_x > right_hand_rel_x else 0

        left_hand_overhead = 1 if points[15][1] < points[11][1] - 0.1 else 0
        right_hand_overhead = 1 if points[16][1] < points[12][1] - 0.1 else 0

        left_hand_side = 1 if abs(left_hand_rel_x) > shoulder_width else 0
        right_hand_side = 1 if abs(right_hand_rel_x) > shoulder_width else 0

        left_arm_spread = abs(left_hand_rel_x) / (shoulder_width + 0.001)
        right_arm_spread = abs(right_hand_rel_x) / (shoulder_width + 0.001)

        # === 2. 動態速度特徵 ===
        # 計算手腕和手肘的移動速度 (dx, dy)
        velocity_features = [0.0] * 8  # 左腕dx,dy, 右腕dx,dy, 左肘dx,dy, 右肘dx,dy

        if self.prev_landmarks_array is not None:
            # 索引: 15=左腕, 16=右腕, 13=左肘, 14=右肘
            indices = [15, 16, 13, 14]
            curr_pts = points[indices, :2]  # 只取 x, y
            prev_pts = self.prev_landmarks_array[indices, :2]

            # 計算位移向量
            diffs = curr_pts - prev_pts
            velocity_features = diffs.flatten().tolist()

        # 更新上一幀記錄
        self.prev_landmarks_array = points.copy()

        # 組合所有特徵
        base_features = [
            left_arm_angle, right_arm_angle,
            left_arm_extension, right_arm_extension,
            left_hand_rel_x, left_hand_rel_y,
            right_hand_rel_x, right_hand_rel_y,
            left_hand_height, right_hand_height,
            hands_height_diff, hands_distance,
            arm_symmetry, body_tilt,
            shoulder_width,
            hands_crossed,
            left_hand_overhead, right_hand_overhead,
            left_hand_side, right_hand_side,
            left_arm_spread, right_arm_spread
        ]

        # 合併靜態與動態特徵
        final_features = np.array(base_features + velocity_features)

        # 處理 NaN
        final_features = np.nan_to_num(final_features)

        return final_features

    def _calculate_angle(self, point1, point2, point3):
        a = np.array([point1[0], point1[1]])
        b = np.array([point2[0], point2[1]])
        c = np.array([point3[0], point3[1]])
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def _calculate_body_tilt(self, left_shoulder, right_shoulder, left_hip, right_hip):
        shoulder_center = (left_shoulder[:2] + right_shoulder[:2]) / 2
        hip_center = (left_hip[:2] + right_hip[:2]) / 2
        dx = shoulder_center[0] - hip_center[0]
        dy = shoulder_center[1] - hip_center[1]
        return np.arctan2(dx, dy)

    def _calculate_arm_extension(self, shoulder, elbow, wrist):
        upper_arm = np.linalg.norm(shoulder[:2] - elbow[:2])
        forearm = np.linalg.norm(elbow[:2] - wrist[:2])
        total_length = upper_arm + forearm
        direct_distance = np.linalg.norm(shoulder[:2] - wrist[:2])
        return direct_distance / (total_length + 1e-6)

    def save_model(self, filepath):
        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'moves': self.moves
        }
        joblib.dump(model_data, filepath)
        print(f"[INFO] 模型已保存到: {filepath}")

    def load_model(self, filepath):
        if os.path.exists(filepath):
            try:
                model_data = joblib.load(filepath)
                self.classifier = model_data['classifier']
                self.scaler = model_data['scaler']
                self.moves = model_data['moves']
                self.is_trained = True
                print(f"[INFO] 模型已從 {filepath} 載入")
                return True
            except Exception as e:
                print(f"[ERROR] 模型載入失敗: {e}")
                return False
        return False

    def get_move_description(self, move_name):
        return self.moves.get(move_name, move_name)
