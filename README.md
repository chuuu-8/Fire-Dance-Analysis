# Fire-Dance-Analysis

```
FireDanceAnalysis/
├── app.py                 # Flask 主應用程式 (負責網站後端與路由)
├── fire_dance_analyzer.py # 火舞分析核心模組 (包含特徵提取與模型預測邏輯)
├── quick_label.py         # 快速數據標註工具 (用來錄製訓練數據)
├── train_model.py         # 模型訓練器 (讀取數據並訓練出 pkl 模型)
├── move_config.json       # 招式設定檔 (定義招式名稱與代碼)
├── requirements.txt       # 依賴套件列表 (列出需要 pip install 的套件)
├── README.md              # 使用說明文件
├── fire_dance_model.pkl   # 訓練好的模型檔案 (由 train_model.py 生成)
│
├── uploads/               # [暫存] 使用者上傳的影片會暫存在此
│
├── static/                # 靜態資源目錄
│   ├── processed/         #    └─ 處理完成的影片 (分析結果)
│   └── style.css          #    └─ 網站外觀樣式表 (深色模式主題)
│
├── templates/             # 網頁模板目錄
│   ├── index.html         #    └─ 首頁 (上傳介面、參數設定)
│   └── result.html        #    └─ 結果頁面 (顯示影片、統計數據)
│
├── training_data/         # 訓練數據目錄
│   └── training_dataset.csv #  └─ 標註好的特徵數據 (由 quick_label.py 生成)
│
├── training_videos/       # 訓練影片原始檔目錄 (放置您的 .mov/.mp4 練習影片)
│
└── outputs/               # 輸出檔案目錄 (備用)
```
