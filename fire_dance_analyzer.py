import cv2
import numpy as np
import mediapipe as mp
import joblib
import os
import json

# 全域招式定義 (Key 必須對應資料夾名稱，Value 是網頁顯示的文字)
# 這裡設定為全英文顯示
ALLOWED_MOVES_DEFAULT = {
    '1-handed spiral wraps': '1-Handed Spiral Wraps',
    '2-beat': '2-Beat',
    '3-beat': '3-Beat',
    '3-beat weave': '3-Beat Weave',
    '3-beat weave with head roll': '3-Beat Weave with Head Roll',
    '3-petal flower': '3-Petal Flower',
    '4-beat': '4-Beat',
    '4-petal': '4-Petal',
    '4-petal flower': '4-Petal Flower',
    '4-petal with isolation': '4-Petal with Isolation',
    'Archer Weave': 'Archer Weave',
    'Butterfly': 'Butterfly',
    'Cap': 'Cap',
    'Continuous toss': 'Continuous Toss',
    'Crosser': 'Crosser',
    'Head roll': 'Head Roll',
    'Isolation': 'Isolation',
    'Other': 'Other',
    'Side 4-petal': 'Side 4-Petal',
    'Stall': 'Stall',
    'Stall chaser': 'Stall Chaser',
    'The superman': 'The Superman'
}


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
        self.moves = ALLOWED_MOVES_DEFAULT.copy()
        self.classifier = None
        self.scaler = None
        self.is_trained = False
        self.prev_landmarks_array = None

    def reset_history(self):
        self.prev_landmarks_array = None

    def load_model(self, filepath):
        if os.path.exists(filepath):
            try:
                data = joblib.load(filepath)
                self.classifier = data['classifier']
                self.scaler = data['scaler']
                # 優先使用模型內儲存的招式表 (如果有的話)
                if 'moves' in data:
                    self.moves = data['moves']
                # 如果模型內沒有 moves，或者您想強制使用上面的英文表，可以把上面那兩行註解掉
                # 但通常建議讓模型帶著自己的招式表走

                self.is_trained = True
                print(f"[INFO] 模型載入成功: {filepath}")
                return True
            except Exception as e:
                print(f"[ERROR] 模型載入失敗: {e}")
        return False

    def extract_pose_features(self, landmarks):
        """提取 30 維特徵 (包含防呆數學計算)"""
        if not landmarks: return None

        # 轉換為 Numpy 陣列
        points = np.array([[lm.x, lm.y, lm.visibility] for lm in landmarks.landmark])

        # --- 靜態特徵 ---
        # 角度計算 (加入 + 1e-6 防止分母為 0)
        left_arm_angle = self._calculate_angle(points[11], points[13], points[15])
        right_arm_angle = self._calculate_angle(points[12], points[14], points[16])

        left_arm_ext = self._calculate_ext(points[11], points[13], points[15])
        right_arm_ext = self._calculate_ext(points[12], points[14], points[16])

        body_center = (points[11][:2] + points[12][:2]) / 2
        l_hand_rel = points[15][:2] - body_center
        r_hand_rel = points[16][:2] - body_center

        l_h, r_h = points[15][1], points[16][1]
        h_diff = abs(l_h - r_h)
        h_dist = np.linalg.norm(points[15][:2] - points[16][:2])

        shoulder_w = np.linalg.norm(points[11][:2] - points[12][:2]) + 1e-6
        arm_symm = abs(left_arm_angle - right_arm_angle) / 180.0
        is_crossed = 1.0 if l_hand_rel[0] > r_hand_rel[0] else 0.0

        l_overhead = 1.0 if points[15][1] < points[11][1] else 0.0
        r_overhead = 1.0 if points[16][1] < points[12][1] else 0.0
        l_side = 1.0 if abs(l_hand_rel[0]) > shoulder_w else 0.0
        r_side = 1.0 if abs(r_hand_rel[0]) > shoulder_w else 0.0

        static_features = [
            left_arm_angle, right_arm_angle, left_arm_ext, right_arm_ext,
            l_hand_rel[0], l_hand_rel[1], r_hand_rel[0], r_hand_rel[1],
            l_h, r_h, h_diff, h_dist, arm_symm, 0.0,
            shoulder_w, is_crossed, l_overhead, r_overhead, l_side, r_side,
            abs(l_hand_rel[0]) / shoulder_w, abs(r_hand_rel[0]) / shoulder_w
        ]

        # --- 動態特徵 ---
        velocity_features = [0.0] * 8
        if self.prev_landmarks_array is not None:
            indices = [15, 16, 13, 14]
            diffs = points[indices, :2] - self.prev_landmarks_array[indices, :2]
            velocity_features = diffs.flatten().tolist()

        self.prev_landmarks_array = points.copy()

        return np.nan_to_num(np.array(static_features + velocity_features))

    def _calculate_angle(self, p1, p2, p3):
        a, b, c = p1[:2], p2[:2], p3[:2]
        ba, bc = a - b, c - b
        # 加上 1e-6 避免分母為 0
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def _calculate_ext(self, s, e, w):
        arm_len = np.linalg.norm(s[:2] - e[:2]) + np.linalg.norm(e[:2] - w[:2])
        direct = np.linalg.norm(s[:2] - w[:2])
        return direct / (arm_len + 1e-6)

    def get_move_description(self, code):
        return self.moves.get(code, code)
