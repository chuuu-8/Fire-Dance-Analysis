from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
import cv2
import yt_dlp
import uuid
import numpy as np
import glob
import json
import datetime
from collections import deque, Counter
from fire_dance_analyzer import FireDanceAnalyzer

# ========== 初始化 Flask 與 資料庫 ==========
app = Flask(__name__)
app.config['SECRET_KEY'] = 'fire-dance-secret-key-888'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///fire_dance.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

analyzer = FireDanceAnalyzer()

# 設定資料夾
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'static/processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# 嘗試載入模型
if os.path.exists('fire_dance_model.pkl'):
    analyzer.load_model('fire_dance_model.pkl')
else:
    print("[WARNING] 未找到 fire_dance_model.pkl，僅能進行骨架偵測")

# ========== 自定義模板過濾器 (將秒數轉為 MM:SS) ==========
@app.template_filter('format_time')
def format_time_filter(seconds):
    if seconds is None: seconds = 0
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"

# ========== 資料庫模型 (Models) ==========

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    records = db.relationship('Record', backref='user', lazy=True)

class Record(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.DateTime, default=datetime.datetime.now)
    video_filename = db.Column(db.String(150))
    original_name = db.Column(db.String(150))
    duration = db.Column(db.Integer)            # 分析的總長度 (秒)
    analysis_start = db.Column(db.Integer, default=0) # 分析開始時間點 (秒)
    analysis_end = db.Column(db.Integer, default=0)   # 分析結束時間點 (秒)
    best_move = db.Column(db.String(50))
    avg_confidence = db.Column(db.Float)
    move_stats_json = db.Column(db.Text)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

# 初始化資料庫
with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ========== 輔助函式 ==========

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

def download_youtube_video(url):
    unique_id = uuid.uuid4().hex[:8]
    safe_name = f"yt_{unique_id}"
    save_path = os.path.join(os.path.abspath(UPLOAD_FOLDER), safe_name)
    ydl_opts = {
        'format': '18/best[ext=mp4]',
        'outtmpl': f'{save_path}.%(ext)s',
        'quiet': False, 'noplaylist': True, 'socket_timeout': 30
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            return filename, info.get('title', 'YouTube Video')
    except Exception as e:
        raise Exception(f"下載失敗: {str(e)}")

def trim_video(input_path, start, end):
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened(): return input_path
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        start_frame = max(0, int(start * fps))
        end_frame = min(int(end * fps), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) if end > 0 else int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if start_frame >= end_frame: return input_path
        output_path = input_path.replace('.', f'_clip_{start}_{end}.')
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (w, h))
        if not out.isOpened(): out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        curr = start_frame
        while cap.isOpened() and curr < end_frame:
            ret, frame = cap.read()
            if not ret: break
            out.write(frame)
            curr += 1
        cap.release();
        out.release()
        return output_path
    except:
        return input_path

# ========== 會員路由 (Auth Routes) ==========

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if User.query.filter_by(username=username).first():
            flash('帳號已存在', 'error')
            return redirect(url_for('register'))
        new_user = User(username=username, password=generate_password_hash(password, method='pbkdf2:sha256'))
        db.session.add(new_user)
        db.session.commit()
        login_user(new_user)
        return redirect(url_for('index'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('index'))
        flash('帳號或密碼錯誤', 'error')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/history')
@login_required
def history():
    records = Record.query.filter_by(user_id=current_user.id).order_by(Record.date.desc()).all()
    for r in records:
        if r.move_stats_json:
            r.stats = json.loads(r.move_stats_json)
        else:
            r.stats = {}
    return render_template('history.html', records=records)

# ========== 主程式路由 ==========

@app.route('/')
def index():
    return render_template('index.html', user=current_user)

@app.route('/api/video_duration', methods=['POST'])
def check_video_duration():
    url = request.form.get('youtube_url')
    if not url: return jsonify({'duration': 100})
    try:
        ydl_opts = {'quiet': True, 'noplaylist': True, 'socket_timeout': 10}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return jsonify({'duration': info.get('duration', 100)})
    except Exception as e:
        print(f"秒數偵測失敗: {e}")
        return jsonify({'duration': 100})

@app.route('/analyze', methods=['POST'])
def analyze():
    input_path = None
    process_path = None
    video_title = "本地上傳影片"

    try:
        url = request.form.get('youtube_url')
        file = request.files.get('file')
        start = int(request.form.get('start', 0))
        end = int(request.form.get('end', 0))
        quality = request.form.get('quality', 'best')
        analysis_mode = request.form.get('analysis_mode', 'balanced')

        if url and url.strip():
            input_path, video_title = download_youtube_video(url.strip())
        elif file and file.filename:
            input_path = os.path.join(UPLOAD_FOLDER, f"local_{uuid.uuid4().hex[:6]}_{file.filename}")
            file.save(input_path)
            video_title = file.filename
        else:
            return render_template('index.html', error="請提供影片", user=current_user)

        process_path = input_path
        if start > 0 or (end > 0 and end > start):
            process_path = trim_video(input_path, start, end)

        output_filename = f"out_{uuid.uuid4().hex[:8]}.mp4"
        output_path = os.path.join(PROCESSED_FOLDER, output_filename)

        cap = cv2.VideoCapture(process_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        target_w = w_orig
        if quality == '720p' and w_orig > 1280: target_w = 1280
        elif quality == '480p' and w_orig > 854: target_w = 854
        elif quality == '360p' and w_orig > 640: target_w = 640
        elif quality == 'best': target_w = min(w_orig, 1280)

        scale = target_w / w_orig
        w, h = int(w_orig * scale), int(h_orig * scale)

        if analysis_mode == 'accurate': analyze_interval = 1
        elif analysis_mode == 'fast': analyze_interval = 5
        else: analyze_interval = 3

        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        if not out.isOpened(): out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        analyzer.reset_history()
        timeline, merged_timeline = [], []
        analyzed_count = 0
        prediction_buffer = deque(maxlen=5)

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1

            if scale != 1.0: frame = cv2.resize(frame, (w, h))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = analyzer.pose.process(rgb)

            if results.pose_landmarks:
                import mediapipe as mp
                mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks,
                                                          analyzer.mp_pose.POSE_CONNECTIONS,
                                                          mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                                                          mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

                if frame_idx % analyze_interval == 0 and analyzer.is_trained:
                    feats = analyzer.extract_pose_features(results.pose_landmarks)
                    if feats is not None:
                        analyzed_count += 1
                        try:
                            probs = analyzer.classifier.predict_proba(analyzer.scaler.transform(feats.reshape(1, -1)))[0]
                            idx = np.argmax(probs)
                            if probs[idx] > 0.3:
                                prediction_buffer.append(analyzer.classifier.classes_[idx])
                            if len(prediction_buffer) >= 3:
                                top_move, count = Counter(prediction_buffer).most_common(1)[0]
                                if count >= 3 and probs[idx] > 0.8:
                                    timeline.append({'time': round(frame_idx / fps, 2),
                                                     'move_zh': analyzer.get_move_description(top_move),
                                                     'confidence': round(float(probs[idx]), 2)})
                        except: pass
            out.write(frame)
        cap.release(); out.release()

        if timeline:
            curr = timeline[0]; curr['start'] = curr['time']; curr['count'] = 1
            for item in timeline[1:]:
                if item['move_zh'] == curr['move_zh'] and (item['time'] - curr['time'] < 1.5):
                    curr['time'] = item['time']; curr['confidence'] = max(curr['confidence'], item['confidence']); curr['count'] += 1
                else:
                    curr['end'] = curr['time']; merged_timeline.append(curr); curr = item; curr['start'] = item['time']; curr['count'] = 1
            curr['end'] = curr['time']; merged_timeline.append(curr)
            merged_timeline = [m for m in merged_timeline if m['count'] >= 3]

        move_stats = {}
        for m in merged_timeline: move_stats[m['move_zh']] = move_stats.get(m['move_zh'], 0) + 1

        best_move = max(move_stats, key=move_stats.get) if move_stats else "無"
        avg_conf = sum(m['confidence'] for m in timeline) / len(timeline) if timeline else 0

        # 計算實際的分析結束時間點
        actual_duration = int(frame_idx / fps)
        actual_end_time = start + actual_duration

        if current_user.is_authenticated:
            new_record = Record(
                video_filename=output_filename,
                original_name=video_title,
                duration=actual_duration,
                analysis_start=start,          # 儲存開始時間
                analysis_end=actual_end_time,  # 儲存結束時間
                best_move=best_move,
                avg_confidence=avg_conf,
                move_stats_json=json.dumps(move_stats),
                user_id=current_user.id
            )
            db.session.add(new_record)
            db.session.commit()

        if input_path and os.path.exists(input_path): os.remove(input_path)
        if process_path and process_path != input_path and os.path.exists(process_path): os.remove(process_path)

        return render_template('result.html',
                               output_video=output_filename,
                               move_timeline=timeline,
                               merged_timeline=merged_timeline,
                               analyzed_frames=analyzed_count,
                               total_frames=frame_idx,
                               move_stats=move_stats,
                               user=current_user)

    except Exception as e:
        return render_template('index.html', error=f"處理失敗: {str(e)}", user=current_user)

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
