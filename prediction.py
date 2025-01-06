import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import os
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.widgets import CheckButtons
import csv

# 設定參數
WINDOW_SIZE = 15  # 15秒的窗口
OVERLAP = 0.8    # 80% 重疊
STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP))  # 滑動步長

def create_sequences_for_prediction(df):
    """為預測創建時間序列窗口"""
    sequences = []
    timestamps = []
    
    # 確保所有需要的列都存在
    required_columns = (
        [f'Channel_{i}_Raw' for i in range(1, 7)] +
        [f'Channel_{i}_Noise' for i in range(1, 7)]
    )
    
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"缺少必要的列: {[col for col in required_columns if col not in df.columns]}")
    
    # 確保數據類型為 float64
    for col in required_columns:
        df[col] = df[col].astype('float64')
    
    # 檢查數值是否有效
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    # 計算特徵
    for i in range(0, len(df) - WINDOW_SIZE + 1, STEP_SIZE):
        window = df.iloc[i:(i + WINDOW_SIZE)]
        timestamps.append(window.index[-1])  # 保存每個窗口的最後一個時間戳
        
        try:
            # 確保數據類型為 float64
            raw_values = window[[f'Channel_{i}_Raw' for i in range(1, 7)]].values.astype('float64')
            noise_values = window[[f'Channel_{i}_Noise' for i in range(1, 7)]].values.astype('float64')
            
            # 使用 np.divide 進行安全的除法運算
            noise_ratios = np.divide(
                noise_values, 
                raw_values, 
                out=np.zeros_like(noise_values, dtype='float64'), 
                where=raw_values!=0
            )
            
            pressure_changes = np.diff(raw_values, axis=0)
            pressure_changes = np.vstack([pressure_changes, pressure_changes[-1]])
            
            pressure_center = np.average(raw_values, axis=1, weights=range(1, 7))
            
            stats_features = np.concatenate([
                np.mean(raw_values, axis=0),
                np.std(raw_values, axis=0),
                np.percentile(raw_values, [25, 50, 75], axis=0).flatten()
            ])
            
            features = np.concatenate([
                raw_values,
                noise_ratios,
                pressure_changes,
                pressure_center.reshape(-1, 1),
                stats_features.reshape(1, -1).repeat(WINDOW_SIZE, axis=0)
            ], axis=1)
            
            sequences.append(features)
            
        except Exception as e:
            print(f"處理窗口 {i} 時發生錯誤: {e}")
            continue
    
    return np.array(sequences), timestamps

def load_latest_model():
    """載入最新的模型"""
    log_dir = '_logs/bed_monitor'
    model_files = [f for f in os.listdir(log_dir) if f.endswith('.keras')]
    
    if not model_files:
        raise FileNotFoundError("找不到任何模型文件")
    
    # 使用文件的修改時間來選擇最新的模型
    latest_model = max(
        model_files,
        key=lambda x: os.path.getmtime(os.path.join(log_dir, x))
    )
    model_path = os.path.join(log_dir, latest_model)

    print(f"模型路徑: {model_path}")
    print(f"正在載入模型: {latest_model}")
    return load_model(model_path)

def calculate_metrics(y_true, y_pred):
    """計算各項評估指標"""
    metrics = {
        '準確率 (Accuracy)': accuracy_score(y_true, y_pred),
        '精確率 (Precision)': precision_score(y_true, y_pred),
        '召回率 (Recall)': recall_score(y_true, y_pred),
        'F1分數': f1_score(y_true, y_pred)
    }
    return metrics

def visualize_predictions(results):
    """視覺化預測結果"""
    # 創建主圖和子圖
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # 繪製實際狀態和預測狀態
    line1, = ax.plot(results.index, results['Actual_Status'], 'b-', label='Actual Status', alpha=0.5)
    line2, = ax.plot(results.index, results['Predicted_Status'], 'r--', label='Predicted Status', alpha=0.7)
    
    # 設置圖表標題和標籤
    ax.set_title('Bed Status Prediction Results', fontsize=14)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Status (0:Empty, 1:Occupied)', fontsize=12)
    
    # 設置 x 軸時間格式
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
    fig.autofmt_xdate()  # 自動調整日期標籤角度
    
    # 設置 y 軸範圍和刻度
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Empty', 'Occupied'])
    
    # 添加網格
    ax.grid(True, alpha=0.3)
    
    # 創建 CheckButtons
    rax = plt.axes([0.02, 0.87, 0.15, 0.1])  # 控制項位置 [left, bottom, width, height]
    check = CheckButtons(
        rax, 
        ['Actual Status', 'Predicted Status'], 
        [True, True]  # 初始狀態都為顯示
    )
    
    def func(label):
        if label == 'Actual Status':
            line1.set_visible(not line1.get_visible())
        elif label == 'Predicted Status':
            line2.set_visible(not line2.get_visible())
        plt.draw()
    
    check.on_clicked(func)
    
    # 調整布局
    plt.tight_layout()
    
    # 保存圖片（注意：保存的圖片不會包含互動控制項）
    plt.savefig('_data/predictions/prediction_visualization.png', dpi=300, bbox_inches='tight')
    print("Prediction visualization saved to: _data/predictions/prediction_visualization.png")
    
    # 顯示圖表
    plt.show()

def save_prediction_metrics(data_file, metrics, predictions_log='_data/predictions/prediction_metrics_log.csv'):
    """保存預測指標到記錄檔"""
    # 準備要記錄的數據
    file_name = os.path.basename(data_file)
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 準備記錄的欄位
    headers = ['Timestamp', 'File_Name', 'Accuracy', 'Precision', 'Recall', 'F1_Score']
    row_data = [
        current_time,
        file_name,
        metrics['準確率 (Accuracy)'],
        metrics['精確率 (Precision)'],
        metrics['召回率 (Recall)'],
        metrics['F1分數']
    ]
    
    # 檢查記錄檔是否存在
    file_exists = os.path.isfile(predictions_log)
    
    # 確保目錄存在
    os.makedirs(os.path.dirname(predictions_log), exist_ok=True)
    
    # 寫入記錄
    with open(predictions_log, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 如果檔案不存在，寫入標題列
        if not file_exists:
            writer.writerow(headers)
        
        # 寫入數據
        writer.writerow(row_data)
    
    print(f"\n預測指標已記錄到: {predictions_log}")

def predict_bed_status(data_file):
    """預測床上狀態並評估準確度"""
    try:
        # 讀取數據
        df = pd.read_csv(data_file)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.set_index('Timestamp', inplace=True)
        
        # 保存原始的 OnBed_Status
        original_status = df['OnBed_Status'].copy()
        
        # 刪除 OnBed_Status 列進行預測
        df.drop(columns=['OnBed_Status'], inplace=True)
        
        # 創建序列
        sequences, timestamps = create_sequences_for_prediction(df)
        
        if len(sequences) == 0:
            raise ValueError("沒有生成任何有效的序列")
        
        # 載入模型並進行預測
        model = load_latest_model()
        predictions = model.predict(sequences)
        predictions = (predictions > 0.5).astype(int)
        
        # 創建結果DataFrame
        results = pd.DataFrame({
            'Timestamp': timestamps,
            'Predicted_Status': predictions.flatten(),
            'Actual_Status': [original_status[timestamp] for timestamp in timestamps]
        })
        results.set_index('Timestamp', inplace=True)
        
        # 計算評估指標
        metrics = calculate_metrics(
            results['Actual_Status'], 
            results['Predicted_Status']
        )
        
        # 保存預測結果
        predictions_dir = os.path.join(os.path.dirname(data_file), 'predictions')
        os.makedirs(predictions_dir, exist_ok=True)
        output_file = os.path.join(
            predictions_dir,
            f"{os.path.splitext(os.path.basename(data_file))[0]}_predictions.csv"
        )
        results.to_csv(output_file)
        
        # 視覺化預測結果
        visualize_predictions(results)
        
        # 保存預測指標到記錄檔
        save_prediction_metrics(data_file, metrics)
        
        # 輸出評估結果
        print(f"\n預測評估結果:")
        for metric_name, score in metrics.items():
            print(f"{metric_name}: {score:.4f}")
        print(f"\n預測完成，結果已保存至: {output_file}")
        
        return results, metrics
        
    except Exception as e:
        print(f"預測過程中發生錯誤: {str(e)}")
        return None, None

if __name__ == "__main__":
    # 示例使用
    data_file = "./_data/SPS2021PA000317_20241229_04_20241230_04_data.csv"
    results, metrics = predict_bed_status(data_file)
