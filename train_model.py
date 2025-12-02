#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from fire_dance_analyzer import FireDanceAnalyzer

class FireDanceModelTrainer:
    def __init__(self, data_dir="training_data"):
        self.data_dir = data_dir
        self.analyzer = FireDanceAnalyzer()
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_score = 0
        
    def load_training_data(self):
        """載入訓練數據"""
        training_file = os.path.join(self.data_dir, "training_dataset.csv")
        
        if not os.path.exists(training_file):
            print(f"訓練數據文件不存在: {training_file}")
            print("請先使用 data_collector.py 收集和標註數據")
            return None
        
        # 載入數據
        df = pd.read_csv(training_file)
        print(f"載入 {len(df)} 條訓練數據")
        
        # 檢查數據分布
        print("\n招式數據分布:")
        move_counts = df['move'].value_counts()
        for move, count in move_counts.items():
            print(f"{self.analyzer.get_move_description(move)}: {count}")
        
        return df
    
    def prepare_features(self, df):
        """準備特徵數據"""
        # 將特徵字符串轉換為數組
        features = []
        for feature_str in df['features']:
            # 移除方括號並分割
            feature_str = feature_str.strip('[]')
            feature_array = [float(x) for x in feature_str.split(',')]
            features.append(feature_array)
        
        X = np.array(features)
        y = df['move'].values
        
        print(f"特徵維度: {X.shape}")
        print(f"標籤數量: {len(y)}")
        
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
        
        print("開始訓練模型...")
        print("-" * 50)
        
        for name, model in models.items():
            print(f"訓練 {name}...")
            
            # 訓練模型
            model.fit(X_train_scaled, y_train)
            
            # 評估模型
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            # 交叉驗證
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            
            results[name] = {
                'model': model,
                'train_score': train_score,
                'test_score': test_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"  訓練準確率: {train_score:.3f}")
            print(f"  測試準確率: {test_score:.3f}")
            print(f"  交叉驗證: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            print()
        
        return results, X_test_scaled, y_test
    
    def select_best_model(self, results):
        """選擇最佳模型"""
        print("模型性能比較:")
        print("-" * 50)
        
        best_model_name = None
        best_score = 0
        
        for name, result in results.items():
            score = result['test_score']
            print(f"{name:20} | 測試準確率: {score:.3f} | 交叉驗證: {result['cv_mean']:.3f}")
            
            if score > best_score:
                best_score = score
                best_model_name = name
        
        print("-" * 50)
        print(f"最佳模型: {best_model_name} (測試準確率: {best_score:.3f})")
        
        self.best_model = results[best_model_name]['model']
        self.best_score = best_score
        
        return best_model_name, results[best_model_name]
    
    def evaluate_model(self, model, X_test, y_test):
        """詳細評估模型"""
        y_pred = model.predict(X_test)
        
        print("\n詳細分類報告:")
        print("-" * 50)
        labels = sorted(list(set(y_test)))
        target_names = [
            self.analyzer.get_move_description(label) if label in self.analyzer.moves else label
            for label in labels
        ]
        print(classification_report(y_test, y_pred, labels=labels, target_names=target_names))
        
        # 混淆矩陣
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=target_names,
                   yticklabels=target_names)
        plt.title('混淆矩陣')
        plt.xlabel('預測標籤')
        plt.ylabel('真實標籤')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return y_pred

    
    def save_model(self, model_name="fire_dance_model.pkl"):
        """保存模型"""
        if self.best_model is None:
            print("沒有可保存的模型")
            return
        
        model_data = {
            'classifier': self.best_model,
            'scaler': self.scaler,
            'moves': self.analyzer.moves,
            'training_info': {
                'best_score': self.best_score,
                'feature_count': self.scaler.n_features_in_,
                'model_type': type(self.best_model).__name__
            }
        }
        
        joblib.dump(model_data, model_name)
        print(f"模型已保存到: {model_name}")
        
        # 同時更新分析器
        self.analyzer.classifier = self.best_model
        self.analyzer.scaler = self.scaler
        self.analyzer.is_trained = True
    
    def generate_training_report(self, results, model_name):
        """生成訓練報告"""
        report_file = "training_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("火舞招式識別模型訓練報告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"訓練時間: {pd.Timestamp.now()}\n")
            f.write(f"最佳模型: {model_name}\n")
            f.write(f"最佳測試準確率: {self.best_score:.3f}\n\n")
            
            f.write("所有模型性能比較:\n")
            f.write("-" * 30 + "\n")
            for name, result in results.items():
                f.write(f"{name}:\n")
                f.write(f"  訓練準確率: {result['train_score']:.3f}\n")
                f.write(f"  測試準確率: {result['test_score']:.3f}\n")
                f.write(f"  交叉驗證: {result['cv_mean']:.3f} (+/- {result['cv_std'] * 2:.3f})\n\n")
            
            f.write("招式定義:\n")
            f.write("-" * 20 + "\n")
            for key, value in self.analyzer.moves.items():
                f.write(f"{key}: {value}\n")
        
        print(f"訓練報告已保存到: {report_file}")
    
    def run_training(self):
        """執行完整的訓練流程"""
        print("火舞招式識別模型訓練")
        print("=" * 50)
        
        # 載入數據
        df = self.load_training_data()
        if df is None:
            return
        
        # 準備特徵
        X, y = self.prepare_features(df)
        
        # 訓練多個模型
        results, X_test, y_test = self.train_multiple_models(X, y)
        
        # 選擇最佳模型
        best_model_name, best_result = self.select_best_model(results)
        
        # 詳細評估
        y_pred = self.evaluate_model(best_result['model'], X_test, y_test)
        
        # 保存模型
        self.save_model()
        
        # 生成報告
        self.generate_training_report(results, best_model_name)
        
        print("\n訓練完成！")
        print(f"最佳模型已保存，測試準確率: {self.best_score:.3f}")

def main():
    """主函數"""
    trainer = FireDanceModelTrainer()
    trainer.run_training()

if __name__ == "__main__":
    main()
