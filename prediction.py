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
<<<<<<< HEAD
=======
from sklearn.preprocessing import StandardScaler
>>>>>>> f93ae05aee297d00757d273e257780a84c8375f2

# 設定參數
WINDOW_SIZE = 15  # 15秒的窗口
OVERLAP = 0.8    # 80% 重疊
STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP))  # 滑動步長

# 添加新的常數定義在文件開頭
WARNING_TIMES = {
    'EARLY': 15,    # 15秒預警
    'IMMEDIATE': 5, # 5秒預警
    'CRITICAL': 2   # 2秒預警
}

def create_sequences_for_prediction(df):
    """為預測創建時間序列窗口"""
    print(f"原始數據形狀: {df.shape}")
    print(f"特徵列: {df.columns.tolist()}")
    
    sequences = []
    timestamps = []
    
    # 只使用原始通道數據（與訓練時一致）
    raw_columns = [f'Channel_{i}_Raw' for i in range(1, 7)]
    
    # 確保所有必要的列都存在
    if not all(col in df.columns for col in raw_columns):
        raise ValueError(f"缺少必要的列: {[col for col in raw_columns if col not in df.columns]}")
    
    # 標準化原始通道數據
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(df[raw_columns])
    df_normalized = pd.DataFrame(normalized_data, columns=raw_columns, index=df.index)
    
    # 創建序列
    for i in range(0, len(df_normalized) - WINDOW_SIZE + 1, STEP_SIZE):
        window = df_normalized.iloc[i:(i + WINDOW_SIZE)]
        timestamps.append(window.index[-1])
        sequences.append(window.values.astype('float64'))
    
    sequences = np.array(sequences)
    print(f"生成序列形狀: {sequences.shape}")
    print(f"時間戳數量: {len(timestamps)}")
    
    return sequences, timestamps

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
<<<<<<< HEAD
=======
    model = load_model(model_path)
    print(f"模型輸入形狀: {model.input_shape}")  # 添加形狀檢查
    print(f"模型輸出形狀: {model.output_shape}")  # 添加形狀檢查
    return model
>>>>>>> f93ae05aee297d00757d273e257780a84c8375f2
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

<<<<<<< HEAD
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
=======
def save_prediction_metrics(data_file, metrics, threshold, predictions_log='_data/predictions/prediction_metrics_log.csv'):
    """保存預測指標到記錄檔"""
    file_name = os.path.basename(data_file)
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    headers = [
        'Timestamp', 'File_Name', 'Threshold',
        'Early_Detections', 'Immediate_Detections', 'Critical_Detections',
        'Missed_Events', 'False_Alarms'
    ]
    
    row_data = [
        current_time,
        file_name,
        f"{threshold:.2f}",
        metrics['early_detections'],
        metrics['immediate_detections'],
        metrics['critical_detections'],
        metrics['missed_events'],
        metrics['false_alarms']
    ]
    
    file_exists = os.path.isfile(predictions_log)
    os.makedirs(os.path.dirname(predictions_log), exist_ok=True)
    
    with open(predictions_log, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
>>>>>>> f93ae05aee297d00757d273e257780a84c8375f2
        writer.writerow(row_data)
    
    print(f"\n預測指標已記錄到: {predictions_log}")

<<<<<<< HEAD
=======
def evaluate_prediction_results(y_true, y_pred, timestamps, find_best_threshold=False):
    """評估預測結果並尋找最佳閾值"""
    # 建議添加
    print(f"評估數據形狀: y_true: {y_true.shape}, y_pred: {y_pred.shape}")
    
    def evaluate_with_threshold(threshold):
        predictions = (y_pred >= threshold).astype(int)
        events_detected = []
        
        # 找出所有預測事件
        i = 0
        while i < len(predictions):
            if predictions[i] == 1:
                start_idx = i
                while i < len(predictions) and predictions[i] == 1:
                    i += 1
                end_idx = i - 1
                
                events_detected.append({
                    'start_time': timestamps[start_idx],
                    'end_time': timestamps[end_idx],
                    'prediction_time': timestamps[start_idx]
                })
            else:
                i += 1
        
        # 找出實際事件
        actual_events = []
        i = 0
        while i < len(y_true):
            if y_true[i] == 1:
                start_idx = i
                while i < len(y_true) and y_true[i] == 1:
                    i += 1
                end_idx = i - 1
                
                actual_events.append({
                    'start_time': timestamps[start_idx],
                    'end_time': timestamps[end_idx],
                    'event_time': timestamps[end_idx]
                })
            else:
                i += 1
        
        # 評估結果
        metrics = {
            'early_detections': 0,     # 提前15秒預測到
            'immediate_detections': 0,  # 提前5秒預測到
            'critical_detections': 0,   # 提前2秒預測到
            'missed_events': 0,        # 漏報
            'false_alarms': 0          # 誤報
        }
        
        # 評估每個實際事件
        for actual_event in actual_events:
            event_detected = False
            best_prediction_time = None
            
            for pred_event in events_detected:
                time_diff = (actual_event['event_time'] - pred_event['prediction_time']).total_seconds()
                
                if 0 < time_diff <= WARNING_TIMES['EARLY']:
                    event_detected = True
                    if best_prediction_time is None or pred_event['prediction_time'] < best_prediction_time:
                        best_prediction_time = pred_event['prediction_time']
            
            if event_detected and best_prediction_time is not None:
                time_diff = (actual_event['event_time'] - best_prediction_time).total_seconds()
                
                if time_diff >= WARNING_TIMES['EARLY']:
                    metrics['early_detections'] += 1
                elif time_diff >= WARNING_TIMES['IMMEDIATE']:
                    metrics['immediate_detections'] += 1
                elif time_diff >= WARNING_TIMES['CRITICAL']:
                    metrics['critical_detections'] += 1
            else:
                metrics['missed_events'] += 1
        
        # 計算誤報
        for pred_event in events_detected:
            matched = False
            for actual_event in actual_events:
                time_diff = (actual_event['event_time'] - pred_event['prediction_time']).total_seconds()
                if 0 < time_diff <= WARNING_TIMES['EARLY']:
                    matched = True
                    break
            if not matched:
                metrics['false_alarms'] += 1
        
        # 計算綜合指標
        total_detections = metrics['early_detections'] + metrics['immediate_detections'] + metrics['critical_detections']
        false_alarm_rate = metrics['false_alarms'] / (total_detections + 1e-6)  # 避免除以零
        detection_rate = total_detections / (total_detections + metrics['missed_events'] + 1e-6)
        
        # 確保即使分數為0也返回有效的metrics
        if total_detections == 0 and metrics['false_alarms'] == 0:
            return metrics, 0.0
        
        # 計算F1-like分數
        f1_score = 2 * (detection_rate * (1 - false_alarm_rate)) / (detection_rate + (1 - false_alarm_rate) + 1e-6)
        return metrics, f1_score

    if find_best_threshold:
        # 測試不同的閾值
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_threshold = 0.5
        best_score = -1
        best_metrics = None
        
        print("\n尋找最佳閾值...")
        for threshold in thresholds:
            metrics, score = evaluate_with_threshold(threshold)
            print(f"閾值: {threshold:.2f}, 分數: {score:.4f}")
            print(f"早期檢測: {metrics['early_detections']}, 誤報: {metrics['false_alarms']}")
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_metrics = metrics
        
        # 確保即使所有分數都是0，也返回有效的metrics
        if best_metrics is None:
            best_metrics, _ = evaluate_with_threshold(0.5)
        
        print(f"\n最佳閾值: {best_threshold:.2f}, 最佳分數: {best_score:.4f}")
        return best_metrics, best_threshold
    else:
        metrics, _ = evaluate_with_threshold(0.5)
        return metrics, 0.5

>>>>>>> f93ae05aee297d00757d273e257780a84c8375f2
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
        print(f"序列形狀: {sequences.shape}")  # 添加形狀檢查
        raw_predictions = model.predict(sequences)
        print(f"預測結果形狀: {raw_predictions.shape}")  # 添加形狀檢查
        
        # 根據訓練結果調整預設閾值
        DEFAULT_THRESHOLD = 0.5  # 因為訓練結果顯示 0.5 是較好的閾值
        
        # 使用預設閾值進行初始預測
        initial_predictions = (raw_predictions > DEFAULT_THRESHOLD).astype(int)
        results = pd.DataFrame({
            'Timestamp': timestamps,
            'Predicted_Status': initial_predictions.flatten(),
            'Actual_Status': [original_status[timestamp] for timestamp in timestamps]
        })
        results.set_index('Timestamp', inplace=True)
        
        # 尋找最佳閾值
        metrics, best_threshold = evaluate_prediction_results(
            results['Actual_Status'].values, 
            results['Predicted_Status'].values,
            results.index
        )
        
<<<<<<< HEAD
        # 保存預測結果
=======
        # 使用最佳閾值更新預測結果
        results['Predicted_Status'] = (raw_predictions > best_threshold).astype(int).flatten()
        
        # 輸出新的評估結果
        print(f"\n使用最佳閾值 {best_threshold:.2f} 的預測評估結果:")
        print(f"提前15秒預測次數: {metrics['early_detections']}")
        print(f"提前5秒預測次數: {metrics['immediate_detections']}")
        print(f"提前2秒預測次數: {metrics['critical_detections']}")
        print(f"漏報次數: {metrics['missed_events']}")
        print(f"誤報次數: {metrics['false_alarms']}")
        
        # 保存預測結果時包含閾值信息
>>>>>>> f93ae05aee297d00757d273e257780a84c8375f2
        predictions_dir = os.path.join(os.path.dirname(data_file), 'predictions')
        os.makedirs(predictions_dir, exist_ok=True)
        output_file = os.path.join(
            predictions_dir,
<<<<<<< HEAD
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
=======
            f"{os.path.splitext(os.path.basename(data_file))[0]}_predictions_threshold_{best_threshold:.2f}.csv"
        )
        results.to_csv(output_file)
        
        # 加回視覺化預測結果的函數調用
        visualize_predictions(results)
        
        # 更新保存預測指標的函數調用
        save_prediction_metrics(data_file, metrics, best_threshold)
        
        print(f"\n預測完成，結果已保存至: {output_file}")
        
        # 添加更多評估指標
        print("\n詳細評估指標:")
        print("基本指標:")
        basic_metrics = calculate_metrics(results['Actual_Status'].values, results['Predicted_Status'].values)
        for metric_name, value in basic_metrics.items():
            print(f"{metric_name}: {value:.4f}")
        
        print("\n時間相關指標:")
        print(f"提前15秒預測次數: {metrics['early_detections']}")
        print(f"提前5秒預測次數: {metrics['immediate_detections']}")
        print(f"提前2秒預測次數: {metrics['critical_detections']}")
        print(f"漏報次數: {metrics['missed_events']}")
        print(f"誤報次數: {metrics['false_alarms']}")
        
        return results, metrics, best_threshold
>>>>>>> f93ae05aee297d00757d273e257780a84c8375f2
        
    except Exception as e:
        print(f"預測過程中發生錯誤: {str(e)}")
        return None, None, None

if __name__ == "__main__":
    # 示例使用
<<<<<<< HEAD
    data_file = "./_data/SPS2021PA000317_20241229_04_20241230_04_data.csv"
    results, metrics = predict_bed_status(data_file)
=======
    data_file = "./_data/SPS2021PA000329_20241215_04_20241216_04_data.csv"
    results, metrics, best_threshold = predict_bed_status(data_file)
>>>>>>> f93ae05aee297d00757d273e257780a84c8375f2
