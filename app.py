from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import os
import cv2
import yt_dlp
import mediapipe as mp
import uuid
import numpy as np
from collections import deque
import re

# ========== 模組導入 ==========
try:
    from fire_dance_analyzer import FireDanceAnalyzer
except ImportError:
    print("⚠️ 警告：無法導入 FireDanceAnalyzer，將使用簡化版分析器")
    FireDanceAnalyzer = None

# ========== Flask 設定 ==========
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'static/processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)


# ========== 工具函數 ==========
def sanitize_filename(filename):
    """清理檔案名稱"""
    if not filename:
        return f"file_{uuid.uuid4().hex[:8]}"

    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    filename = re.sub(r'\s+', '_', filename.strip())
    filename = filename.strip('.')

    if len(filename) > 100:
        name, ext = os.path.splitext(filename)
        filename = name[:100 - len(ext)] + ext

    if not filename or filename in ['.', '_', '-']:
        filename = f"file_{uuid.uuid4().hex[:8]}"

    return filename


def enhanced_detect_fire_in_frame(frame):
    """改進的火頭檢測 - 平衡精確度和檢測率"""
    results = []
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    fire_ranges = [
        ([0, 100, 100], [15, 255, 255]),
        ([15, 100, 100], [35, 255, 255]),
        ([160, 100, 100], [180, 255, 255])
    ]

    fire_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in fire_ranges:
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        fire_mask = cv2.bitwise_or(fire_mask, mask)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bright_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)[1]

    final_mask = cv2.bitwise_or(fire_mask, bright_mask)

    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_small)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_medium)

    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        area = cv2.contourArea(contour)

        if 30 < area < 5000:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

                frame_h, frame_w = frame.shape[:2]
                if cx < 20 or cx > frame_w - 20 or cy < 20 or cy > frame_h - 20:
                    continue

                rect = cv2.minAreaRect(contour)
                width, height = rect[1]

                if width > 0 and height > 0:
                    aspect_ratio = max(width, height) / min(width, height)
                    perimeter = cv2.arcLength(contour, True)
                    circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0

                    if aspect_ratio > 1.5:
                        tool_type = 'stick'
                        confidence = min(0.9, area / 500)
                    elif circularity > 0.5:
                        tool_type = 'ball'
                        confidence = min(0.9, area / 400)
                    else:
                        tool_type = 'auto'
                        confidence = min(0.8, area / 600)

                    if confidence > 0.2:
                        results.append({
                            'position': (cx, cy),
                            'type': tool_type,
                            'area': area,
                            'confidence': confidence
                        })

                        if len(results) >= 3:
                            break

    return results


def smooth_trajectory(trajectory, window=5):
    """軌跡平滑化"""
    trajectory = list(trajectory)
    if len(trajectory) < window:
        return trajectory
    smoothed = []
    for i in range(len(trajectory)):
        start = max(0, i - window // 2)
        end = min(len(trajectory), i + window // 2 + 1)
        avg_point = np.mean(trajectory[start:end], axis=0)
        smoothed.append(tuple(map(int, avg_point)))
    return smoothed


def analyze_move_advanced(trajectory, pose_landmarks=None):
    """改進的動作分析"""
    if len(trajectory) < 5:
        return "準備中", 0.0

    points = np.array(trajectory)
    centroid = np.mean(points, axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)

    avg_distance = np.mean(distances)
    trajectory_spread = np.std(distances)

    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    width, height = max_x - min_x, max_y - min_y

    velocities = np.diff(points, axis=0)
    speeds = np.linalg.norm(velocities, axis=1)
    avg_speed = np.mean(speeds)

    angles = np.arctan2(velocities[:, 1], velocities[:, 0])
    angle_changes = np.abs(np.diff(angles))
    direction_changes = np.sum(angle_changes > np.pi / 6)

    try:
        hull = cv2.convexHull(points.astype(np.float32))
        area = cv2.contourArea(hull)
        perimeter = cv2.arcLength(hull, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
    except:
        circularity = 0

    total_length = np.sum(speeds)
    straight_distance = np.linalg.norm(points[-1] - points[0])
    linearity = straight_distance / total_length if total_length > 0 else 0

    move_name, confidence = "未知動作", 0.0

    if circularity > 0.6 and trajectory_spread < 80:
        move_name = "風車" if avg_distance > 80 else "POI旋轉"
        confidence = min(1.0, circularity * 0.8)
    elif 0.2 < circularity < 0.9 and direction_changes > 2:
        move_name, confidence = "8字形", min(1.0, 0.3 + (direction_changes / 8) * 0.7)
    elif height > 0 and (width / height) > 1.5 and linearity > 0.4:
        move_name, confidence = "水平掃動", min(1.0, linearity * 0.6)
    elif width > 0 and (height / width) > 1.5 and linearity > 0.4:
        move_name, confidence = "垂直掃動", min(1.0, linearity * 0.6)
    elif direction_changes > 4 and 0.3 < circularity < 0.8:
        move_name, confidence = "蝴蝶", min(1.0, 0.2 + (direction_changes / 12) * 0.8)
    elif trajectory_spread < 50 and direction_changes > 5:
        move_name, confidence = "分離", min(1.0, 0.3 + (direction_changes / 15) * 0.7)
    elif width > 0 and height > 0 and (width / height) > 1.2 and direction_changes > 2:
        move_name, confidence = "無限符號", min(1.0, 0.3 + ((width / height) / 2.5) * 0.7)
    elif avg_speed > 20:
        move_name, confidence = "快速移動", min(1.0, avg_speed / 100)
    elif direction_changes > 3:
        move_name, confidence = "旋轉", min(1.0, direction_changes / 10)

    return move_name, confidence


MOVE_NAMES_EN = {
    'horizontal_sweep': 'Horizontal Sweep',
    'vertical_sweep': 'Vertical Sweep',
    'figure_8': 'Figure 8',
    'infinity': 'Infinity',
    'poi_spin': 'POI Spin',
    'butterfly': 'Butterfly',
    'windmill': 'Windmill',
    'isolation': 'Isolation',
    '水平掃動': 'Horizontal Sweep',
    '垂直掃動': 'Vertical Sweep',
    '8字形': 'Figure 8',
    '無限符號': 'Infinity',
    'POI旋轉': 'POI Spin',
    '蝴蝶': 'Butterfly',
    '風車': 'Windmill',
    '分離': 'Isolation',
    '快速移動': 'Fast Movement',
    '旋轉': 'Rotation',
    '準備中': 'Preparing',
    '未知動作': 'Unknown Move'
}


def merge_move_timeline(move_timeline, min_duration=0.0):
    """將連續相同的動作合併成時間段（包含所有動作）"""
    if not move_timeline:
        return []

    merged_segments = []
    current_segment = None

    for entry in move_timeline:
        move = entry.get('move')
        move_en = entry.get('move_en')
        time = entry.get('time')
        confidence = entry.get('confidence', 0.0)
        tool_type = entry.get('tool_type', 'auto')

        # 如果是新動作或第一個動作
        if current_segment is None or current_segment['move'] != move:
            if current_segment:
                merged_segments.append(current_segment)

            current_segment = {
                'start': time,
                'end': time,
                'move': move,
                'move_en': move_en,
                'confidence': confidence,
                'tool_type': tool_type,
                'count': 1
            }
        else:
            # 延續當前片段
            current_segment['end'] = time
            current_segment['confidence'] = (
                                                    current_segment['confidence'] * current_segment[
                                                'count'] + confidence
                                            ) / (current_segment['count'] + 1)
            current_segment['count'] += 1

    # 保存最後一個片段
    if current_segment:
        merged_segments.append(current_segment)

    # 添加持續時間（不過濾，保留所有片段）
    for seg in merged_segments:
        duration = seg['end'] - seg['start']
        seg['duration'] = round(duration, 2)

    return merged_segments


def download_youtube_video(youtube_url, quality):
    """下載 YouTube 影片"""
    try:
        safe_filename = f"youtube_{uuid.uuid4().hex[:8]}"
        safe_filename = re.sub(r'[^A-Za-z0-9_]', '_', safe_filename)

        format_map = {
            "best": "bestvideo+bestaudio/best",
            "720p": "bestvideo[height<=720]+bestaudio/best[height<=720]/best[height<=720]",
            "480p": "bestvideo[height<=480]+bestaudio/best[height<=480]/best[height<=480]",
        }
        format_selector = format_map.get(quality, "bestvideo+bestaudio/best")

        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        uploads_abs = os.path.abspath(UPLOAD_FOLDER)
        output_template = os.path.join(uploads_abs, f'{safe_filename}.%(ext)s').replace('\\', '/')

        ydl_opts = {
            'format': format_selector,
            'outtmpl': output_template,
            'quiet': True,
            'no_warnings': True,
            'noplaylist': True,
            'restrictfilenames': True,
            'windowsfilenames': True,
            'merge_output_format': 'mp4',
        }

        print(f"開始下載 YouTube 影片: {youtube_url}")

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=True)
            candidate = ydl.prepare_filename(info_dict)
            dir_to_check = uploads_abs
            base_prefix = os.path.splitext(os.path.basename(candidate))[0]
            found_path = None
            if candidate and os.path.exists(candidate):
                found_path = candidate
            else:
                for fname in os.listdir(dir_to_check):
                    if fname.startswith(base_prefix + '.'):
                        found_path = os.path.join(dir_to_check, fname)
                        break

            if found_path and os.path.exists(found_path):
                print(f"影片下載成功: {found_path}")
                return os.path.abspath(found_path)
            raise Exception("下載的檔案不存在")

    except Exception as e:
        print(f"YouTube 下載錯誤: {e}")
        raise Exception(f"YouTube 影片下載失敗: {str(e)}")


def trim_video(input_path, output_path, start, end):
    """裁剪影片"""
    try:
        input_path = os.path.abspath(input_path)
        output_path = os.path.abspath(output_path)
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise Exception(f"無法開啟影片檔案: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            raise Exception("無法獲取影片總幀數")

        start_frame = max(0, int(start * fps))
        if end > 0:
            end_frame = min(int(end * fps), total_frames)
        else:
            end_frame = total_frames

        if start_frame >= end_frame:
            raise Exception("開始時間必須小於結束時間")

        codecs = ['mp4v', 'XVID', 'avc1']
        out = None

        for codec in codecs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                if codec == 'XVID':
                    output_path = output_path.replace('.mp4', '.avi')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                if out.isOpened():
                    print(f"使用編碼格式: {codec}")
                    break
            except Exception as e:
                print(f"編碼 {codec} 失敗: {e}")
                continue

        if out is None or not out.isOpened():
            raise Exception("無法創建影片寫入器")

        frame_count = 0
        written_frames = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count >= start_frame and frame_count < end_frame:
                out.write(frame)
                written_frames += 1

            frame_count += 1
            if frame_count >= end_frame:
                break

        cap.release()
        out.release()

        if written_frames == 0:
            raise Exception("沒有寫入任何幀")

        print(f"影片裁剪完成: {written_frames} 幀")
        return output_path

    except Exception as e:
        print(f"影片裁剪錯誤: {e}")
        raise


def process_video_enhanced(input_path, output_path, analysis_mode='balanced'):
    """改進版影片處理函數 - 固定每 0.5 秒偵測一次"""
    print(f"開始分析影片: {input_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise Exception(f"無法開啟影片檔案: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30

    DETECTION_INTERVAL = 0.5
    sample_rate = max(1, int(fps * DETECTION_INTERVAL))

    print(f"影片資訊: {width}x{height}, FPS={fps:.1f}")
    print(f"偵測間隔: {DETECTION_INTERVAL} 秒 (每 {sample_rate} 幀分析一次)")

    if analysis_mode == 'fast':
        max_resolution = 480
    elif analysis_mode == 'accurate':
        max_resolution = 1080
    else:
        max_resolution = 720

    if width > max_resolution:
        scale_factor = max_resolution / width
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
    else:
        new_width, new_height = width, height
        scale_factor = 1.0

    codecs = ['avc1', 'mp4v', 'XVID']
    out = None
    actual_output_path = output_path

    for codec in codecs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            if codec == 'XVID':
                actual_output_path = output_path.replace('.mp4', '.avi')
            else:
                actual_output_path = output_path

            out = cv2.VideoWriter(actual_output_path, fourcc, fps, (new_width, new_height))
            if out.isOpened():
                print(f"使用編碼格式: {codec}")
                break
        except Exception as e:
            print(f"編碼 {codec} 失敗: {e}")
            continue

    if out is None or not out.isOpened():
        raise Exception("無法創建影片寫入器")

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    frame_count = 0
    analyzed_frames = 0
    move_timeline = []

    fire_trajectory = deque(maxlen=30)
    trajectory_timestamps = deque(maxlen=30)
    move_history = deque(maxlen=10)
    detection_history = deque(maxlen=3)
    stable_detection = None

    print(f"開始處理 {total_frames} 幀影片...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_time = frame_count / fps

        stable_move = "準備中"
        avg_confidence = 0.0
        tool_type = "auto"

        if scale_factor != 1.0:
            frame = cv2.resize(frame, (new_width, new_height))

        if frame_count % sample_rate == 0:
            analyzed_frames += 1

            current_detections = enhanced_detect_fire_in_frame(frame) or []
            detection_history.append(current_detections)

            if len(detection_history) >= 2:
                recent_detections = list(detection_history)[-2:]
                detection_counts = {}

                for detections in recent_detections:
                    for det in detections:
                        pos_key = f"{det['position'][0] // 30}_{det['position'][1] // 30}"
                        if pos_key not in detection_counts:
                            detection_counts[pos_key] = []
                        detection_counts[pos_key].append(det)

                best_key = None
                best_count = 0
                for key, dets in detection_counts.items():
                    if len(dets) >= 1:
                        avg_conf = sum(d['confidence'] for d in dets) / len(dets)
                        if len(dets) > best_count or (len(dets) == best_count and avg_conf > 0.3):
                            best_key = key
                            best_count = len(dets)

                if best_key:
                    stable_detections = detection_counts[best_key]
                    stable_detection = max(stable_detections, key=lambda x: x['confidence'])
                else:
                    stable_detection = None
            else:
                stable_detection = current_detections[0] if current_detections else None

            if stable_detection:
                fire_trajectory.append(stable_detection['position'])
                trajectory_timestamps.append(current_time)

                pos = stable_detection['position']
                fire_type = stable_detection['type']
                confidence = stable_detection['confidence']

                color = (0, 255, 255) if fire_type == 'stick' else (255, 0, 255)
                cv2.circle(frame, pos, 8, color, -1)
                cv2.circle(frame, pos, 12, (255, 255, 255), 2)

            if len(fire_trajectory) > 3 and len(trajectory_timestamps) > 3:
                trajectory_list = list(fire_trajectory)
                timestamp_list = list(trajectory_timestamps)

                current_time_threshold = current_time - 3.0
                visible_indices = [i for i, t in enumerate(timestamp_list) if t >= current_time_threshold]

                if len(visible_indices) > 2:
                    visible_trajectory = [trajectory_list[i] for i in visible_indices]
                    visible_timestamps = [timestamp_list[i] for i in visible_indices]

                    smoothed_trajectory = smooth_trajectory(visible_trajectory, window=3)

                    for i in range(1, len(smoothed_trajectory)):
                        start_pos = smoothed_trajectory[i - 1]
                        end_pos = smoothed_trajectory[i]

                        time_diff = current_time - visible_timestamps[i]
                        alpha = max(0.2, 1.0 - (time_diff / 3.0))

                        thickness = max(2, int(4 + alpha * 3))
                        color_intensity = int(255 * alpha)

                        if alpha > 0.7:
                            color = (0, color_intensity, color_intensity)
                        elif alpha > 0.4:
                            color = (0, color_intensity, int(color_intensity * 0.8))
                        else:
                            color = (0, int(color_intensity * 0.6), int(color_intensity * 0.4))

                        cv2.line(frame, start_pos, end_pos, color, thickness)

                        if i % 2 == 0:
                            cv2.circle(frame, end_pos, max(1, int(thickness / 2)), color, -1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(rgb_frame)

            if len(fire_trajectory) > 5:
                current_move, confidence = analyze_move_advanced(
                    list(fire_trajectory), pose_results.pose_landmarks
                )

                move_history.append((current_move, confidence))

                if len(move_history) >= 1:
                    recent_slice = list(move_history)[-1:]
                    recent_moves = [move for move, conf in recent_slice]
                    move_counts = {}
                    for move in recent_moves:
                        move_counts[move] = move_counts.get(move, 0) + 1

                    stable_move = max(move_counts, key=move_counts.get)
                    avg_confidence = np.mean([conf for move, conf in recent_slice
                                              if move == stable_move])

                    if stable_detection:
                        tool_type = stable_detection['type']

            num_fire = 1 if stable_detection else 0
            if not (stable_move == '準備中' and num_fire == 0):
                move_timeline.append({
                    'time': current_time,
                    'move': stable_move,
                    'move_en': MOVE_NAMES_EN.get(stable_move, stable_move),
                    'confidence': avg_confidence,
                    'fire_detections': num_fire,
                    'tool_type': tool_type
                })

        out.write(frame)

        if frame_count % 100 == 0:
            progress = frame_count / total_frames * 100
            print(f"已處理 {frame_count}/{total_frames} 幀 ({progress:.1f}%)")

    cap.release()
    out.release()
    pose.close()

    print(f"\n影片分析完成！")
    print(f"- 總幀數: {total_frames}")
    print(f"- 分析幀數: {analyzed_frames}")
    print(f"- 實際偵測間隔: {DETECTION_INTERVAL} 秒")
    print(f"- 識別動作: {len(move_timeline)} 個")

    return analyzed_frames, total_frames, move_timeline, actual_output_path


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        youtube_url = request.form.get('youtube_url')
        file = request.files.get('file')
        quality = request.form.get('quality', 'best')
        analysis_mode = request.form.get('analysis_mode', 'balanced')
        start = int(request.form.get('start', 0))
        end = int(request.form.get('end', 0))

        input_path = None

        if youtube_url and youtube_url.strip():
            try:
                input_path = download_youtube_video(youtube_url.strip(), quality)
                print(f"YouTube 影片下載成功: {input_path}")
            except Exception as e:
                return f"YouTube 下載失敗: {str(e)}", 400
        elif file and file.filename:
            try:
                clean_filename = sanitize_filename(file.filename)
                safe_filename = f"upload_{uuid.uuid4().hex[:8]}_{clean_filename}"
                input_path = os.path.join(UPLOAD_FOLDER, safe_filename)

                input_path = os.path.abspath(input_path)
                os.makedirs(os.path.dirname(input_path), exist_ok=True)

                file.save(input_path)
                print(f"檔案上傳成功: {input_path}")
            except Exception as e:
                print(f"檔案上傳錯誤: {str(e)}")
                return f"檔案上傳失敗: {str(e)}", 400
        else:
            return "請提供影片檔案或 YouTube 連結", 400

        if not input_path or not os.path.exists(input_path):
            return "輸入檔案不存在", 400

        output_id = str(uuid.uuid4())
        clip_path = os.path.join(PROCESSED_FOLDER, f'clip_{output_id}.mp4')
        output_filename = f'output_{output_id}.mp4'
        output_path = os.path.join(PROCESSED_FOLDER, output_filename)

        os.makedirs(PROCESSED_FOLDER, exist_ok=True)

        try:
            actual_clip_path = trim_video(input_path, clip_path, start, end)
            print(f"影片裁剪完成: {actual_clip_path}")
        except Exception as e:
            return f"影片裁剪失敗: {str(e)}", 400

        try:
            analyzed, total, timeline, final_output_path = process_video_enhanced(
                actual_clip_path, output_path, analysis_mode
            )
            print(f"影片分析完成: {output_path}")
        except Exception as e:
            return f"影片分析失敗: {str(e)}", 400

        if os.path.exists(actual_clip_path):
            os.remove(actual_clip_path)
        if input_path != actual_clip_path and os.path.exists(input_path):
            os.remove(input_path)

        move_stats = {}
        tool_stats = {}

        for move_info in timeline:
            move_zh = move_info.get('move_zh', move_info.get('move_en', move_info.get('move', '未知動作')))
            tool_type = move_info.get('tool_type', 'unknown')
            move_stats[move_zh] = move_stats.get(move_zh, 0) + 1
            tool_stats[tool_type] = tool_stats.get(tool_type, 0) + 1

        # 合併時間段
        merged_timeline = merge_move_timeline(timeline, min_duration=0.5)

        # 打印到控制台
        print("\n動作時間段分析:")
        print("=" * 60)
        for seg in merged_timeline:
            print(
                f"{seg['start']:.2f}-{seg['end']:.2f}  {seg['move']}  (信心度: {seg['confidence'] * 100:.0f}%, 持續: {seg['duration']:.2f}s)")
        print("=" * 60)

        return render_template('result.html',
                               total_frames=total,
                               analyzed_frames=analyzed,
                               output_video=os.path.basename(final_output_path),
                               move_timeline=timeline,
                               merged_timeline=merged_timeline,
                               move_stats=move_stats,
                               tool_stats=tool_stats,
                               analysis_mode=analysis_mode)

    except Exception as e:
        print(f"分析過程發生錯誤: {e}")
        return f"分析失敗: {str(e)}", 500


@app.route('/api/moves', methods=['GET'])
def get_moves():
    """Get all supported moves list"""
    moves = MOVE_NAMES_EN
    return jsonify(moves)


@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)


if __name__ == '__main__':
    print("Fire Dance Analysis System Starting...")
    print("Supported Moves:")
    for eng_name, en_name in MOVE_NAMES_EN.items():
        if eng_name not in ['準備中', '未知動作']:
            print(f"  - {en_name}")

    print(f"\nUpload Folder: {UPLOAD_FOLDER}")
    print(f"Output Folder: {PROCESSED_FOLDER}")

    if FireDanceAnalyzer is not None:
        print("✅ FireDanceAnalyzer loaded")
    else:
        print("⚠️  Using built-in analyzer")

    print("System ready!")

    app.run(debug=True, host='0.0.0.0', port=5000)