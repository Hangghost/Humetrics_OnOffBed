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
from sklearn.preprocessing import StandardScaler

# 設定參數
WINDOW_SIZE = 15  # 15秒的窗口
OVERLAP = 0.8    # 80% 重疊
STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP))  # 滑動步長

# 修改預警時間設定
WARNING_TIME = 10  # 設定單一預警時間（秒）

# 設定路徑
LOG_DIR = "./_logs/bed_monitor_test_sum"
PREDICTIONS_DIR = "./_data/predictions"
DATA_DIR = "./_data/odd"

# 確保必要的目錄存在
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

FIND_BEST_THRESHOLD = False
SUM_ONLY = True

def create_sequences_for_prediction(df, use_sum_only=SUM_ONLY):
    """為預測創建時間序列窗口，增加use_sum_only參數"""
    print(f"原始數據形狀: {df.shape}")
    print(f"特徵列: {df.columns.tolist()}")
    
    sequences = []
    timestamps = []
    
    if use_sum_only:
        # 計算總和並創建新的DataFrame
        df['pressure_sum'] = df[[f'Channel_{i}_Raw' for i in range(1, 7)]].sum(axis=1)
        df_features = pd.DataFrame({'pressure_sum': df['pressure_sum']})
    else:
        # 使用原始通道數據
        raw_columns = [f'Channel_{i}_Raw' for i in range(1, 7)]
        if not all(col in df.columns for col in raw_columns):
            raise ValueError(f"缺少必要的列: {[col for col in raw_columns if col not in df.columns]}")
        df_features = df[raw_columns].copy()
    
    # 標準化數據
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(df_features)
    df_normalized = pd.DataFrame(normalized_data, columns=df_features.columns, index=df.index)
    
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
    log_dir = LOG_DIR
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
    model = load_model(model_path)
    print(f"模型輸入形狀: {model.input_shape}")  # 添加形狀檢查
    print(f"模型輸出形狀: {model.output_shape}")  # 添加形狀檢查
    return model

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
    visualization_path = os.path.join(PREDICTIONS_DIR, 'prediction_visualization.png')
    plt.savefig(visualization_path, dpi=300, bbox_inches='tight')
    print(f"Prediction visualization saved to: {visualization_path}")
    
    # 顯示圖表
    plt.show()

def save_prediction_metrics(data_file, metrics, threshold, predictions_log='_data/predictions/prediction_metrics_log.csv'):
    """保存預測指標到記錄檔"""
    file_name = os.path.basename(data_file)
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    headers = [
        'Timestamp', 'File_Name', 'Threshold',
        'Detections', 'Missed_Events', 'False_Alarms'
    ]
    
    row_data = [
        current_time,
        file_name,
        f"{threshold:.2f}",
        metrics['detections'],
        metrics['missed_events'],
        metrics['false_alarms']
    ]
    
    file_exists = os.path.isfile(predictions_log)
    os.makedirs(os.path.dirname(predictions_log), exist_ok=True)
    
    with open(predictions_log, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(row_data)
    
    print(f"\n預測指標已記錄到: {predictions_log}")

def evaluate_prediction_results(y_true, y_pred, timestamps, find_best_threshold=False):
    """評估預測結果並尋找最佳閾值"""
    print(f"評估數據形狀: y_true: {y_true.shape}, y_pred: {y_pred.shape}")
    
    def evaluate_with_threshold(threshold):
        predictions = (y_pred >= threshold).astype(int)
        events_detected = []
        
        # 找出所有預測事件
        i = 0
        while i < len(predictions):
            if predictions[i] == 1:
                # 記錄整個預警時間段
                pred_start = timestamps[i]
                while i < len(predictions) and predictions[i] == 1:
                    i += 1
                pred_end = timestamps[min(i, len(timestamps)-1)]
                
                events_detected.append({
                    'start_time': pred_start,
                    'end_time': pred_end,
                    'prediction_time': pred_end  # 使用結束時間作為預測時間
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
            'detections': 0,     # 成功預測
            'missed_events': 0,  # 漏報
            'false_alarms': 0    # 誤報
        }
        
        # 評估每個實際事件
        for actual_event in actual_events:
            event_detected = False
            for pred_event in events_detected:
                # 檢查預測時間段是否覆蓋了實際事件的預警時間段
                pred_period = (pred_event['start_time'], pred_event['end_time'])
                actual_warning_start = actual_event['event_time'] - timedelta(seconds=WARNING_TIME)
                
                if (pred_period[0] <= actual_event['event_time'] and 
                    pred_period[1] >= actual_warning_start):
                    event_detected = True
                    break
            
            if event_detected:
                metrics['detections'] += 1
            else:
                metrics['missed_events'] += 1
        
        # 計算誤報
        for pred_event in events_detected:
            matched = False
            for actual_event in actual_events:
                time_diff = (actual_event['event_time'] - pred_event['prediction_time']).total_seconds()
                if 0 < time_diff <= WARNING_TIME:
                    matched = True
                    break
            if not matched:
                metrics['false_alarms'] += 1
        
        # 計算綜合指標
        total_detections = metrics['detections']
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
            print(f"檢測: {metrics['detections']}, 誤報: {metrics['false_alarms']}")
            
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

def predict_bed_status(data_file, use_sum_only=SUM_ONLY):
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
        
        # 創建序列，傳入use_sum_only參數
        sequences, timestamps = create_sequences_for_prediction(df, use_sum_only=SUM_ONLY)
        
        if len(sequences) == 0:
            raise ValueError("沒有生成任何有效的序列")
        
        # 載入模型並進行預測
        model = load_latest_model()
        print(f"序列形狀: {sequences.shape}")
        raw_predictions = model.predict(sequences)
        print(f"預測結果形狀: {raw_predictions.shape}")
        
        # 根據訓練結果調整預設閾值
        DEFAULT_THRESHOLD = 0.8  # 因為訓練結果顯示 0.5 是較好的閾值
        
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
        
        # 使用最佳閾值更新預測結果
        results['Predicted_Status'] = (raw_predictions > best_threshold).astype(int).flatten()
        
        # 輸出新的評估結果
        print(f"\n使用最佳閾值 {best_threshold:.2f} 的預測評估結果:")
        print(f"成功預測次數: {metrics['detections']}")
        print(f"漏報次數: {metrics['missed_events']}")
        print(f"誤報次數: {metrics['false_alarms']}")
        
        # 保存預測結果時包含閾值信息
        predictions_dir = os.path.join(os.path.dirname(data_file), 'predictions')
        os.makedirs(predictions_dir, exist_ok=True)
        output_file = os.path.join(
            predictions_dir,
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
        print(f"成功預測次數: {metrics['detections']}")
        print(f"漏報次數: {metrics['missed_events']}")
        print(f"誤報次數: {metrics['false_alarms']}")
        
        return results, metrics, best_threshold
        
    except Exception as e:
        print(f"預測過程中發生錯誤: {str(e)}")
        return None, None, None

if __name__ == "__main__":
    # 示例使用
    data_file = os.path.join(DATA_DIR, "SPS2021PA000329_20250201_04_20250202_04_data.csv")
    # 設置是否使用總和值
    use_sum_only = SUM_ONLY
    results, metrics, best_threshold = predict_bed_status(data_file, use_sum_only=use_sum_only)
