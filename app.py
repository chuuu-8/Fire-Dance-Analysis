from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import os
import cv2
import yt_dlp
import mediapipe as mp
import uuid
import numpy as np
from collections import deque
import re
import json
import joblib

# ========== 模組導入 ==========
try:
    from fire_dance_analyzer import FireDanceAnalyzer
except ImportError:
    print("[WARNING] 無法導入 FireDanceAnalyzer")
    FireDanceAnalyzer = None

# ========== Flask 設定 ==========
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'static/processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# === 招式定義 ===
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
    'isolation': 'isolation',
    'other': '其他',
    'preparing': '準備中'
}


# ========== 載入招式列表 ==========
def load_move_names():
    return ALLOWED_MOVES_DEFAULT.copy()


MOVES_DICT = load_move_names()
MOVE_NAMES_EN = {}
for k, v in MOVES_DICT.items():
    MOVE_NAMES_EN[k] = k
    MOVE_NAMES_EN[v] = k


# ========== 工具函數 ==========
def sanitize_filename(filename):
    if not filename: return f"file_{uuid.uuid4().hex[:8]}"
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    filename = re.sub(r'\s+', '_', filename.strip())
    if len(filename) > 100:
        name, ext = os.path.splitext(filename)
        filename = name[:100 - len(ext)] + ext
    return filename if filename else f"file_{uuid.uuid4().hex[:8]}"


def get_video_duration(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return 100
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 100
        cap.release()
        return int(duration)
    except:
        return 100


def download_youtube_video(youtube_url, quality):
    """下載 YouTube 影片 (更新版：自動重試與容錯)"""
    try:
        safe_filename = f"youtube_{uuid.uuid4().hex[:8]}"
        # 設定絕對路徑
        output_path = os.path.join(os.path.abspath(UPLOAD_FOLDER), f'{safe_filename}.%(ext)s')

        print(f"[INFO] 開始下載 YouTube 影片: {youtube_url}")

        # 策略 1: 嘗試標準最佳單檔 (best)
        # 這裡加上 ignoreerrors 和 nocheckcertificate 以應對各種網路/憑證問題
        ydl_opts = {
            'format': 'best',
            'outtmpl': output_path,
            'quiet': True,
            'noplaylist': True,
            'ignoreerrors': True,
            'no_check_certificate': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            # 如果下載失敗，info 可能是 None
            if not info:
                raise Exception("無法獲取影片資訊 (請確認 yt-dlp 已更新)")

            filename = ydl.prepare_filename(info)

            if os.path.exists(filename) and os.path.getsize(filename) > 0:
                print(f"[INFO] 影片下載成功: {filename}")
                return filename

            # 搜尋可能的副檔名
            base = os.path.splitext(filename)[0]
            for ext in ['.mp4', '.webm', '.mkv', '.3gp']:
                if os.path.exists(base + ext) and os.path.getsize(base + ext) > 0:
                    return base + ext

            # 如果 best 失敗，嘗試 worst (保底策略)
            print("[WARNING] 最佳畫質下載失敗，嘗試保底模式...")
            ydl_opts['format'] = 'worst'
            with yt_dlp.YoutubeDL(ydl_opts) as ydl_retry:
                info = ydl_retry.extract_info(youtube_url, download=True)
                filename = ydl_retry.prepare_filename(info)
                if os.path.exists(filename): return filename

            raise Exception("所有下載嘗試均失敗")

    except Exception as e:
        print(f"[ERROR] YouTube 下載錯誤: {e}")
        # 這裡會提示用戶去更新
        if "Signature extraction failed" in str(e) or "Requested format is not available" in str(e):
            raise Exception("下載工具過期，請執行: pip install --upgrade yt-dlp")
        raise Exception(f"YouTube 下載失敗: {str(e)}")


def trim_video(input_path, output_path, start, end):
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened(): raise Exception("無法開啟影片")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        start_frame = max(0, int(start * fps))
        end_frame = min(int(end * fps), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) if end > 0 else int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if start_frame >= end_frame: start_frame, end_frame = 0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        codecs = ['avc1', 'h264', 'mp4v', 'XVID']
        out = None

        for codec in codecs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                temp_out = output_path.replace('.mp4', '.avi') if codec == 'XVID' else output_path
                out = cv2.VideoWriter(temp_out, fourcc, fps, (width, height))
                if out.isOpened():
                    output_path = temp_out
                    break
            except:
                continue

        if not out or not out.isOpened(): raise Exception("編碼失敗")

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if start_frame <= frame_count < end_frame: out.write(frame)
            frame_count += 1
            if frame_count >= end_frame: break

        cap.release()
        out.release()
        return output_path
    except Exception as e:
        raise


def enhanced_detect_fire_in_frame(frame, pose_landmarks=None):
    results = []
    if pose_landmarks is None: return [], None, None

    h, w = frame.shape[:2]
    lm = pose_landmarks.landmark
    shoulder_width = abs(lm[11].x - lm[12].x) * w
    search_radius = max(int(shoulder_width * 6.0), 150)

    left_wrist = (int(lm[15].x * w), int(lm[15].y * h))
    right_wrist = (int(lm[16].x * w), int(lm[16].y * h))
    valid_zones = [left_wrist, right_wrist]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    fire_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    fire_ranges = [([0, 100, 100], [15, 255, 255]), ([15, 100, 100], [35, 255, 255]),
                   ([160, 100, 100], [180, 255, 255]), ([0, 0, 200], [180, 50, 255])]
    for lower, upper in fire_ranges:
        fire_mask = cv2.bitwise_or(fire_mask, cv2.inRange(hsv, np.array(lower), np.array(upper)))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
    final_mask = cv2.bitwise_or(fire_mask, bright_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        area = cv2.contourArea(contour)
        if 20 < area < 20000:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                point_vec = np.array([cx, cy])
                min_dist = min([np.linalg.norm(point_vec - np.array(wrist)) for wrist in valid_zones])
                if min_dist <= search_radius:
                    results.append({'position': (cx, cy), 'confidence': 1.0})
                    if len(results) >= 2: break
    return results, search_radius, valid_zones


def merge_move_timeline(move_timeline, min_duration=0.0):
    if not move_timeline: return []
    merged = []
    current = None
    for entry in move_timeline:
        move_zh = entry.get('move_zh')
        if current is None or current['move_zh'] != move_zh:
            if current: merged.append(current)
            current = {
                'start': entry['time'], 'end': entry['time'],
                'move': entry['move'], 'move_zh': move_zh,
                'confidence': entry['confidence'], 'count': 1
            }
        else:
            current['end'] = entry['time']
            current['confidence'] = (current['confidence'] * current['count'] + entry['confidence']) / (
                        current['count'] + 1)
            current['count'] += 1
    if current: merged.append(current)
    for seg in merged: seg['duration'] = round(seg['end'] - seg['start'], 2)
    return merged


def process_video_enhanced(input_path, output_path, analysis_mode='balanced'):
    print(f"[INFO] 分析影片: {input_path}")

    # === 直球對決模式設定 ===
    CONF_THRESH = 0.15  # 門檻

    analyzer = None
    if FireDanceAnalyzer:
        analyzer = FireDanceAnalyzer()
        analyzer.reset_history()
        if os.path.exists('fire_dance_model.pkl'):
            analyzer.load_model('fire_dance_model.pkl')
            print("[INFO] 模型載入成功")
        else:
            print("[WARNING] 找不到 fire_dance_model.pkl")

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 確保 total_frames 存在 (修復之前的 Bug)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    scale_factor = 720 / width if width > 720 else 1.0
    new_w, new_h = int(width * scale_factor), int(height * scale_factor)

    codecs = ['avc1', 'h264', 'mp4v', 'XVID']
    out = None
    for c in codecs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*c)
            temp_out = output_path.replace('.mp4', '.avi') if c == 'XVID' else output_path
            out = cv2.VideoWriter(temp_out, fourcc, fps, (new_w, new_h))
            if out.isOpened():
                output_path = temp_out
                break
        except:
            continue

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3)

    frame_count = 0
    analyzed_frames = 0
    move_timeline = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        current_time = frame_count / fps

        frame_resized = cv2.resize(frame, (new_w, new_h)) if scale_factor != 1.0 else frame
        rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(rgb_frame)
        landmarks = pose_results.pose_landmarks

        detections, radius, valid_zones = enhanced_detect_fire_in_frame(frame_resized, landmarks)

        if landmarks:
            mp_drawing.draw_landmarks(
                frame_resized, landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            )
        else:
            cv2.putText(frame_resized, "No Pose", (20, new_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        for d in detections:
            cv2.circle(frame_resized, d['position'], 5, (0, 165, 255), -1)

        # 動作識別
        if frame_count % 3 == 0:
            analyzed_frames += 1

            if landmarks and analyzer and analyzer.is_trained:
                try:
                    features = analyzer.extract_pose_features(landmarks)
                    if features is not None:
                        features_scaled = analyzer.scaler.transform(features.reshape(1, -1))
                        probs = analyzer.classifier.predict_proba(features_scaled)[0]
                        idx = np.argmax(probs)
                        code = analyzer.classifier.classes_[idx]
                        conf = probs[idx]

                        move_name = analyzer.get_move_description(code)
                        if code == 'other': move_name = "其他"

                        # 只要有信心度就顯示
                        if conf > CONF_THRESH:
                            move_timeline.append({
                                'time': round(current_time, 2),
                                'move': code,
                                'move_zh': move_name,
                                'confidence': round(float(conf), 2)
                            })

                            debug_text = f"{move_name} {int(conf * 100)}%"
                            color = (0, 255, 0) if code != 'other' else (0, 0, 255)
                            cv2.putText(frame_resized, debug_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                except Exception as e:
                    print(e)

        out.write(frame_resized)

    cap.release()
    out.release()
    pose.close()
    return analyzed_frames, total_frames, move_timeline, output_path


@app.route('/')
def index(): return render_template('index.html')


@app.route('/api/video_duration', methods=['POST'])
def check_video_duration():
    try:
        url = request.form.get('youtube_url')
        if url:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                return jsonify({'duration': info.get('duration', 100)})
    except:
        pass
    return jsonify({'duration': 100})


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        url = request.form.get('youtube_url')
        file = request.files.get('file')
        quality = request.form.get('quality', 'best')
        mode = request.form.get('analysis_mode', 'balanced')
        start = int(request.form.get('start', 0))
        end = int(request.form.get('end', 0))

        if url and url.strip():
            input_path = download_youtube_video(url.strip(), quality)
        elif file and file.filename:
            path = os.path.join(UPLOAD_FOLDER, sanitize_filename(file.filename))
            file.save(path)
            input_path = path
        else:
            return "無輸入", 400

        total_dur = get_video_duration(input_path)
        if end == 0 or end > total_dur: end = total_dur

        output_name = f'out_{uuid.uuid4().hex[:8]}.mp4'
        output_path = os.path.join(PROCESSED_FOLDER, output_name)

        process_path = input_path
        if start > 0 or end < total_dur:
            clip_path = os.path.join(PROCESSED_FOLDER, f'clip_{uuid.uuid4()}.mp4')
            try:
                process_path = trim_video(input_path, clip_path, start, end)
            except:
                pass

        analyzed, total, timeline, final_path = process_video_enhanced(process_path, output_path, mode)

        if process_path != input_path and os.path.exists(process_path): os.remove(process_path)
        if os.path.exists(input_path): os.remove(input_path)

        merged = merge_move_timeline(timeline)
        stats = {}
        for m in timeline: stats[m['move_zh']] = stats.get(m['move_zh'], 0) + 1

        return render_template('result.html', analyzed_frames=analyzed, total_frames=total,
                               output_video=os.path.basename(final_path), move_timeline=timeline,
                               merged_timeline=merged, move_stats=stats)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", 500


@app.route('/processed/<filename>')
def processed_file(filename): return send_from_directory(PROCESSED_FOLDER, filename)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
