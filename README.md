```
FireDanceAnalysis/
├── .idea/                 # PyCharm 編輯器設定檔目錄 (自動生成，不影響程式)
├── .venv/                 # Python 虛擬環境 (包含安裝好的套件，如 OpenCV, Flask 等)
├── downloads/             # (備用/暫存) 下載資料夾
├── instance/              # Flask 資料庫預設目錄
│   └── fire_dance.db      #   └─ SQLite 資料庫 (儲存會員帳號密碼與歷史紀錄)
├── output/                # (備用/舊版) 輸出資料夾
├── outputs/               # (備用/舊版) 輸出資料夾
├── processed/             # 處理完成的影片暫存區 (分析完的骨架影片)
├── static/                # 靜態資源目錄
│   └── style.css          #   └─ 網站外觀樣式表 (深色主題、排版設定)
├── templates/             # 網頁模板目錄 (所有看到的網頁畫面都在這)
│   ├── history.html       #   └─ 歷史紀錄頁
│   ├── index.html         #   └─ 首頁 (上傳影片、選擇畫質/模式)
│   ├── login.html         #   └─ 會員登入頁
│   ├── register.html      #   └─ 會員註冊頁
│   └── result.html        #   └─ 分析結果頁 (顯示播放器與各招式數據)
├── training_data/         # 訓練數據目錄
│   └── training_dataset.csv # └─ 執行 train_model 後，從影片自動抽出的骨架特徵表
├── training_videos/       # 訓練影片原始檔 (餵給 AI 學習的教材)
├── uploads/               # [暫存] 使用者上傳或 YT 下載的影片 (分析完後會刪除)
│
├── app.py                 # Flask 主程式 (負責路由、資料庫存取、呼叫分析器)
├── confusion_matrix.png   # 訓練圖表
├── fire_dance_analyzer.py # 火舞分析核心 (定義 22 個招式、特徵提取與預測邏輯)
├── fire_dance_model.pkl   # 訓練好的AI
├── move_config.json       # [舊版] 招式設定檔 (目前已整合進 analyzer 中)
├── openh264-1.8.0-win64.dll # OpenCV 處理 mp4 影片的解碼器依賴檔
├── quick_label.py         # [舊版] 數據標註工具 (已被自動提取取代)
├── train_model.py         # 模型訓練器 (會去讀 training_videos 並生出新模型)
└── training_report.txt    # 訓練結果文字報告
```
