import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, LSTM, BatchNormalization, Multiply, Bidirectional, Add, Concatenate, UpSampling1D
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from tensorflow.keras.layers import Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import sys
import matplotlib as mpl
import matplotlib.font_manager as fm
import argparse
import glob

# 查找系統上支援中文的字型
chinese_fonts = ['Arial Unicode MS', 'Microsoft YaHei', 'SimHei', 'SimSun', 'Heiti TC', 'STHeiti', 'PingFang TC', 'PingFang HK', 'Hiragino Sans GB']

# 嘗試設定字型
font_found = False
for font in chinese_fonts:
    try:
        mpl.rcParams['font.family'] = font
        plt.rcParams['axes.unicode_minus'] = False  # 正確顯示負號
        font_found = True
        print(f"使用字型: {font}")
        break
    except:
        continue

if not font_found:
    print("警告: 未找到支援中文的字型，圖表中的中文可能無法正確顯示")

# 設定隨機種子
np.random.seed(1337)

# 修改預警時間設定
WARNING_TIME = 15  # 設定單一預警時間（秒）

# 修改INPUT_DATA_PATH的定義和相關導入
INPUT_DATA_DIR = "./_data/pyqt_viewer/training"
INPUT_DATA_PATTERN = "*_data.csv"
# 待實際執行時才獲取檔案清單
INPUT_DATA_PATHS = []  # 先設為空，執行時填入

# 新增預測資料夾路徑
PREDICTION_DATA_DIR = "./_data/training/prediction"
PREDICTION_DATA_PATTERN = "*_data.csv"
PREDICTION_DATA_PATHS = []  # 預測檔案清單

TRAINING_LOG_PATH = "training_test_sum.csv"
FINAL_MODEL_PATH = "final_model_test_sum.keras"
TRAINING_HISTORY_PATH = "training_history_test_sum.png"
LOG_DIR = "./_logs/bed_monitor_test_sum"

# 全域變數聲明
PROCESSED_DATA_PATH = ""
APPLY_BALANCING = False
POS_TO_NEG_RATIO = 0.05
FIND_BEST_THRESHOLD = False

# 確保LOG_DIR和其他必要目錄存在
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs("./_data/training", exist_ok=True)

SILENCE_TIME = 0


# 自定義的評估回調
class EventEvaluationCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, timestamps):
        super().__init__()
        self.validation_data = validation_data
        self.timestamps = timestamps
        self.best_metrics = None
        self.early_detection_history = []
        self.late_detection_history = []
        self.early_time_diff_history = []
        self.false_alarm_history = []
        self.threshold = 0.3  # 降低閾值從0.9到0.3
        self.target_lead_time = 7  # 目標提前時間（秒）
        
    def on_epoch_end(self, epoch, logs=None):
        x_val, y_val = self.validation_data
        y_pred = self.model.predict(x_val)
        
        # 調整預測的形狀，將(samples, 1)壓平為(samples,)
        y_pred_flat = y_pred.flatten()
        
        metrics, threshold, target_lead = evaluate_predictions(y_val, y_pred_flat, self.timestamps, find_best_threshold=False)
        
        if metrics is not None:
            # 計算平均提前時間
            avg_early_time = np.mean(metrics['early_time_diffs']) if metrics['early_time_diffs'] else 0
            
            # 計算時間準確率（使用秒數範圍）
            if metrics['early_time_diffs']:
                time_diff_seconds = abs(avg_early_time - target_lead)
                if time_diff_seconds <= 3:  # 在目標時間±3秒範圍內
                    time_accuracy = 100.0
                else:
                    # 超出範圍，根據秒數差距計算準確性（百分比）
                    time_accuracy = max(0, 100.0 * (1.0 - (time_diff_seconds - 3) / 7))
            else:
                time_accuracy = 0.0
            
            # 記錄指標歷史
            self.early_detection_history.append(metrics['early_detections'])
            self.late_detection_history.append(metrics['late_detections'])
            self.early_time_diff_history.append(avg_early_time)
            self.false_alarm_history.append(metrics['false_alarms'])
            
            print(f"\nEpoch {epoch + 1} - 評估指標 (閾值={threshold:.2f}):")
            print(f"總檢測: {metrics['detections']} (提前: {metrics['early_detections']}, 延遲: {metrics['late_detections']})")
            print(f"漏報: {metrics['missed_events']}, 誤報: {metrics['false_alarms']}")
            print(f"平均提前時間: {avg_early_time:.2f} 秒 (目標: {target_lead:.1f} 秒)")
            print(f"時間準確率: {time_accuracy:.2f}% (±3秒內為100%)")
            
            # 將這些指標添加到日誌中
            logs['early_detections'] = metrics['early_detections']
            logs['late_detections'] = metrics['late_detections']
            logs['missed_events'] = metrics['missed_events']
            logs['false_alarms'] = metrics['false_alarms']
            logs['avg_early_time'] = avg_early_time
            logs['time_accuracy'] = time_accuracy
        else:
            print(f"\nEpoch {epoch + 1} - 無法獲取有效指標")
    
    def on_train_end(self, logs=None):
        # 訓練結束時，繪製指標歷史圖表
        if self.early_detection_history:
            epochs = range(1, len(self.early_detection_history) + 1)
            
            plt.figure(figsize=(12, 10))
            
            # 繪製提前檢測與延遲檢測歷史
            plt.subplot(3, 1, 1)
            plt.plot(epochs, self.early_detection_history, 'g-', label='提前檢測')
            plt.plot(epochs, self.late_detection_history, 'r-', label='延遲檢測')
            plt.title('訓練過程中的檢測類型')
            plt.xlabel('Epoch')
            plt.ylabel('檢測次數')
            plt.legend()
            
            # 繪製平均提前時間歷史
            plt.subplot(3, 1, 2)
            plt.plot(epochs, self.early_time_diff_history, 'b-', label='平均提前時間')
            plt.axhline(y=self.target_lead_time, color='r', linestyle='--', label=f'目標提前時間 ({self.target_lead_time}秒)')
            # 添加目標範圍區域
            plt.axhspan(self.target_lead_time - 3, self.target_lead_time + 3, alpha=0.2, color='green', label='目標範圍 (±3秒)')
            plt.title('訓練過程中的平均提前時間')
            plt.xlabel('Epoch')
            plt.ylabel('時間（秒）')
            plt.legend()
            
            # 繪製誤報歷史
            plt.subplot(3, 1, 3)
            plt.plot(epochs, self.false_alarm_history, 'r-', label='誤報')
            plt.title('訓練過程中的誤報率')
            plt.xlabel('Epoch')
            plt.ylabel('誤報次數')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(LOG_DIR, 'time_prediction_metrics.png'))
            plt.close()

def get_cleaned_data_path(raw_data_path):
    """根據原始數據路徑生成清理後數據的路徑"""
    # 獲取原始檔案名（不含路徑）
    raw_filename = os.path.basename(raw_data_path)
    # 在檔名前加上 'cleaned_' 前綴
    cleaned_filename = f"cleaned_{raw_filename}"
    # 組合完整路徑
    cleaned_data_path = os.path.join("./_data/training", cleaned_filename)
    
    # 確保目錄存在
    os.makedirs(os.path.dirname(cleaned_data_path), exist_ok=True)
    
    return cleaned_data_path

def save_processed_sequences(sequences, labels, cleaned_data_path, feature_names, event_binary=None):
    """保存處理後的序列資料，使用完整時間序列格式"""
    global PROCESSED_DATA_PATH
    # 將sequences轉換回DataFrame，使用原始欄位名稱
    df = pd.DataFrame(sequences, columns=feature_names)
    PROCESSED_DATA_PATH = cleaned_data_path.replace('.csv', '_processed.csv')
    
    # 保存CSV格式，使用原始欄位名稱
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    # 將 .csv 副檔名改為 .npz
    sequences_path = cleaned_data_path.replace('.csv', '_processed.npz')
    
    # 檢查是否提供了 event_binary
    if event_binary is not None:
        np.savez(sequences_path, 
                sequences=sequences, 
                labels=event_binary,  # 直接使用 event_binary 作為標籤
                event_binary=event_binary,
                feature_names=feature_names)  # 使用保存的欄位名稱
    else:
        np.savez(sequences_path, 
                sequences=sequences, 
                labels=labels,
                event_binary=labels,
                feature_names=feature_names)  # 使用保存的欄位名稱
    
    print(f"已保存npz: {sequences_path}")

def detect_bed_events(df):
    """
    檢測離床事件，並在事件發生後的靜默時間內不檢測新事件。
    在原始 DataFrame 中添加 event_binary 欄位，值為 0 或 1
    
    當SILENCE_TIME=0時，將檢測所有1->0的離床事件
    當SILENCE_TIME>0時，在事件發生後的靜默時間內不檢測新事件
    
    返回:
    - 新增了 event_binary 欄位的 DataFrame
    """
    events = []
    event_binary = np.zeros(len(df))  # 創建一個全0陣列，長度與原始資料相同
    status_changes = df['OnBed_Status'].diff()
    last_event_time = -SILENCE_TIME  # 初始化為負值，確保第一個事件可以被檢測
    
    # 找出所有離床事件
    for idx in range(1, len(df)):
        # 檢查是否在靜默時間內（只有當SILENCE_TIME>0時才進行檢查）
        if SILENCE_TIME > 0 and idx - last_event_time < SILENCE_TIME:
            continue
            
        # 只檢測 1->0 的離床事件
        if status_changes.iloc[idx] == -1:  # 1->0 離床
            events.append({
                'time': idx,
                'type': 'leaving',
                'original_status': 1
            })
            # 只在離床事件的確切時間點標記為1
            event_binary[idx] = 1
            last_event_time = idx  # 更新最後事件時間

    print(f"檢測到 {len(events)} 個離床事件")
    
    # 將 event_binary 添加到原始 DataFrame
    df['event_binary'] = event_binary
    
    return df

def create_sequences(df, cleaned_data_path, apply_balancing=APPLY_BALANCING, pos_to_neg_ratio=POS_TO_NEG_RATIO):
    """修改後的序列創建函數，使用固定長度的時間序列"""
    # 增強特徵工程
    df['Raw_sum'] = df[[f'Channel_{i}_Raw' for i in range(1, 7)]].sum(axis=1)
    df['Noise_max'] = df[[f'Channel_{i}_Noise' for i in range(1, 7)]].max(axis=1)
    
    # 移除不需要的欄位
    noise_columns = [col for col in df.columns if col.startswith('Channel_') and col.endswith('_Noise')]
    max_columns = [col for col in df.columns if col.startswith('Channel_') and col.endswith('_Max')]
    columns_to_drop = noise_columns + max_columns
    
    if columns_to_drop:
        print(f"正在移除以下欄位: {columns_to_drop}")
        df = df.drop(columns=columns_to_drop)
    
    # 填充NaN值
    df = df.ffill().bfill()

    print(f'df.columns: {df.columns}')
    
    # 檢測事件並添加 event_binary 欄位
    df = detect_bed_events(df)

    # 修改特徵列表，移除OnBed_Status和event_binary
    input_feature_columns = [f'Channel_{i}_Raw' for i in range(1, 7)]
    input_feature_columns.append('Raw_sum')
    input_feature_columns.append('Noise_max')
    
    # 檢查必要列是否存在（用於檢查，但不全部用於特徵）
    required_columns = input_feature_columns.copy()
    required_columns.append('OnBed_Status')
    required_columns.append('event_binary')
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"缺少必要的列: {[col for col in required_columns if col not in df.columns]}")
    
    # 只複製實際用於模型的特徵列
    df_features = df[input_feature_columns].copy()
    
    # 存一份CSV（包含所有列，便於檢查）
    df.to_csv(cleaned_data_path, index=False)
    print(f"已保存CSV: {cleaned_data_path}")
    
    # 保存實際用於模型的特徵名稱
    feature_names = df_features.columns.tolist()
    
    # 轉換為數組時保留欄位名稱資訊
    sequences = df_features.values.astype('float64')
    labels = df['event_binary'].values
    
    print(f"============序列長度: {len(sequences)}")

    # 確保序列長度為86400
    if len(sequences) < 86401:
        # 使用零填充
        padding = np.zeros((86401 - len(sequences), sequences.shape[1]))
        sequences = np.vstack([sequences, padding])
        labels = np.pad(labels, (0, 86401 - len(labels)), 'constant')
        print(f"已填充序列長度: {len(sequences)}")

    elif len(sequences) > 86401:
        # 直接截斷超過的部分
        print(f"序列長度超過86400，直接截斷多餘部分")
        sequences = sequences[:86401]
        labels = labels[:86401]
        print(f"已截斷序列長度: {len(sequences)}")
        print(f"已截斷序列標籤長度: {len(labels)}")
    
    # 保存處理後的數據
    save_processed_sequences(sequences, labels, cleaned_data_path, feature_names, df['event_binary'].values)
    
    return sequences, labels, df['event_binary'].values, feature_names

def load_and_process_data(raw_data_path, apply_balancing=APPLY_BALANCING, pos_to_neg_ratio=POS_TO_NEG_RATIO):
    # 獲取對應的清理後數據路徑
    cleaned_data_path = get_cleaned_data_path(raw_data_path)

    sequences_path = cleaned_data_path.replace('.csv', '_processed.npz')
    
    # 檢查是否存在已處理的序列資料
    if os.path.exists(sequences_path):
        try:
            data = np.load(sequences_path)
            sequences = data['sequences']
            labels = data['labels']
            event_binary = data['event_binary']
            feature_names = data['feature_names']
            print(f"發現已處理的序列資料，直接讀取: {sequences_path}")
            print(f"特徵名稱: {feature_names}")
            return sequences, labels, event_binary, feature_names
        except Exception as e:
            print(f"讀取序列資料時發生錯誤: {e}")
            print("將重新處理原始數據...")
    
    # 讀取原始數據
    try:
        dataset = pd.read_csv(raw_data_path)
        print(f"數據集形狀: {dataset.shape}")
        print(f"數據集列: {dataset.columns.tolist()}")
        
        # 檢查是否已經是處理過的數據（已經有Noise_max和Raw_sum欄位）
        if 'Noise_max' in dataset.columns and 'Raw_sum' in dataset.columns:
            print("檢測到已處理過的數據檔案，跳過特徵工程步驟")
            
            # 修改特徵列表，僅包含八個特徵
            input_feature_columns = [f'Channel_{i}_Raw' for i in range(1, 7)]
            input_feature_columns.extend(['Raw_sum', 'Noise_max'])
            
            # 確保有必要的列（用於檢查）
            required_columns = input_feature_columns.copy()
            required_columns.append('OnBed_Status')
            
            # 檢查是否有event_binary欄位，沒有則添加
            if 'event_binary' not in dataset.columns:
                print("未找到event_binary欄位，將生成該欄位")
                dataset = detect_bed_events(dataset)
            
            required_columns.append('event_binary')
            
            if not all(col in dataset.columns for col in required_columns):
                missing_cols = [col for col in required_columns if col not in dataset.columns]
                raise ValueError(f"缺少必要的列: {missing_cols}")
            
            # 只選取實際用於模型的特徵列
            df_features = dataset[input_feature_columns].copy()
            
            # 保存實際用於模型的特徵名稱
            feature_names = df_features.columns.tolist()
            
            # 轉換為數組
            sequences = df_features.values.astype('float64')
            labels = dataset['event_binary'].values
            
            print(f"============序列長度: {len(sequences)}")
            
            # 確保序列長度符合要求
            if len(sequences) < 86401:
                # 使用零填充
                padding = np.zeros((86401 - len(sequences), sequences.shape[1]))
                sequences = np.vstack([sequences, padding])
                labels = np.pad(labels, (0, 86401 - len(labels)), 'constant')
                print(f"已填充序列長度: {len(sequences)}")
            elif len(sequences) > 86401:
                # 直接截斷超過的部分
                print(f"序列長度超過86400，直接截斷多餘部分")
                sequences = sequences[:86401]
                labels = labels[:86401]
                print(f"已截斷序列長度: {len(sequences)}")
            
            # 保存處理後的數據
            save_processed_sequences(sequences, labels, cleaned_data_path, feature_names, dataset['event_binary'].values)
            
            return sequences, labels, dataset['event_binary'].values, feature_names
        else:
            # 原始數據需要完整處理
            sequences, labels, event_binary, feature_names = create_sequences(
                dataset, 
                cleaned_data_path, 
                apply_balancing=apply_balancing, 
                pos_to_neg_ratio=pos_to_neg_ratio
            )
            return sequences, labels, event_binary, feature_names
    except Exception as e:
        print(f"數據處理錯誤: {e}")
        raise

def evaluate_predictions(y_true, y_pred, timestamps, find_best_threshold=FIND_BEST_THRESHOLD):
    """修改後的評估函數，專注於時間差異評估，適應漸進式標籤"""
    def evaluate_with_threshold(threshold):
        # 創建二值預測結果
        predictions = (y_pred >= threshold).astype(int)
        
        # 找出所有預測事件 - 連續的預測為1的段落被視為一個事件
        predicted_events = []
        i = 0
        while i < len(predictions):
            if predictions[i] == 1:
                start_idx = i
                while i < len(predictions) and predictions[i] == 1:
                    i += 1
                end_idx = i - 1
                
                # 儲存事件開始和結束時間
                predicted_events.append({
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'start_time': timestamps[start_idx],
                    'end_time': timestamps[end_idx]
                })
            else:
                i += 1
        
        # 找出所有實際事件 - 標籤超過閾值（對於漸進式標籤，通常是0.7）被視為離床事件
        actual_events = []
        i = 0
        high_threshold = 0.7
        while i < len(y_true):
            if y_true[i] >= high_threshold:
                start_idx = i
                # 找出這個事件的最高標籤點（事件發生時刻）
                max_label_idx = start_idx
                max_label = y_true[start_idx]
                
                while i < len(y_true) and y_true[i] >= 0.5:  # 尋找連續的事件區域
                    if y_true[i] > max_label:
                        max_label = y_true[i]
                        max_label_idx = i
                    i += 1
                end_idx = i - 1
                
                # 儲存事件信息
                actual_events.append({
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'peak_idx': max_label_idx,  # 事件最高點（通常是離床時刻）
                    'start_time': timestamps[start_idx],
                    'end_time': timestamps[end_idx],
                    'peak_time': timestamps[max_label_idx]
                })
            else:
                i += 1
        
        # 擴展的評估指標
        metrics = {
            'detections': 0,       # 成功預測
            'early_detections': 0, # 提前預測
            'late_detections': 0,  # 延遲預測
            'missed_events': 0,    # 漏報
            'false_alarms': 0,     # 誤報
            'time_diffs': [],      # 時間差距（負值表示提前，正值表示延遲）
            'early_time_diffs': [], # 提前預測的時間差距
            'late_time_diffs': []   # 延遲預測的時間差距
        }
        
        # 評估每個實際事件
        for actual_event in actual_events:
            event_detected = False
            best_time_diff = float('inf')
            best_pred_event = None
            
            # 檢查所有預測事件
            for pred_event in predicted_events:
                # 計算預測開始時間與實際事件發生時間的差距
                time_diff = pred_event['start_time'] - actual_event['peak_time']
                
                # 我們認為以下情況是有效的檢測：
                # 1. 預測在實際事件前WARNING_TIME秒內開始（提前預測）
                # 2. 預測在實際事件後5秒內開始（輕微延遲）
                if -WARNING_TIME <= time_diff <= 5:
                    event_detected = True
                    # 選擇時間差距最小的預測
                    if abs(time_diff) < abs(best_time_diff):
                        best_time_diff = time_diff
                        best_pred_event = pred_event
            
            # 記錄檢測結果
            if event_detected and best_pred_event is not None:
                metrics['detections'] += 1
                metrics['time_diffs'].append(best_time_diff)
                
                # 分類為提前或延遲預測
                if best_time_diff <= 0:
                    metrics['early_detections'] += 1
                    metrics['early_time_diffs'].append(abs(best_time_diff))
                else:
                    metrics['late_detections'] += 1
                    metrics['late_time_diffs'].append(best_time_diff)
                
                # 將此預測事件標記為已匹配
                best_pred_event['matched'] = True
            else:
                metrics['missed_events'] += 1
        
        # 計算誤報 - 未能與任何實際事件匹配的預測
        for pred_event in predicted_events:
            if not pred_event.get('matched', False):
                metrics['false_alarms'] += 1
        
        # 計算評估分數
        
        # 檢測率 = 成功檢測 / 總實際事件數
        detection_rate = metrics['detections'] / max(len(actual_events), 1)
        
        # 提前預測率 = 提前預測 / 總檢測數
        early_detection_rate = metrics['early_detections'] / max(metrics['detections'], 1)
        
        # 平均提前時間（秒）
        avg_early_time = np.mean(metrics['early_time_diffs']) if metrics['early_time_diffs'] else 0
        
        # 誤報率 = 誤報 / 總預測事件數
        false_alarm_rate = metrics['false_alarms'] / max(len(predicted_events), 1)
        
        # 計算綜合分數，考慮以下因素：
        # 1. 檢測率 - 越高越好
        # 2. 提前預測率 - 越高越好
        # 3. 平均提前時間 - 接近目標提前時間(秒)最佳
        # 4. 誤報率 - 越低越好
        
        # 目標提前時間（秒）
        target_lead_time = 7  # 目標為提前7秒預測
        
        # 時間預測準確性（使用固定秒數誤差範圍）
        # 如果平均提前時間在目標時間的±3秒範圍內，則視為完全準確
        # 否則根據距離目標時間的秒數差距計算準確性
        if metrics['early_time_diffs']:
            time_diff_seconds = abs(avg_early_time - target_lead_time)
            if time_diff_seconds <= 3:  # 在目標時間±3秒範圍內
                time_accuracy = 1.0
            else:
                # 超出範圍，根據秒數差距計算準確性
                # 最多允許差距10秒，超過10秒則準確性為0
                time_accuracy = max(0, 1.0 - (time_diff_seconds - 3) / 7)
        else:
            time_accuracy = 0.0
        
        # 綜合分數計算
        score = (0.4 * detection_rate) + (0.3 * early_detection_rate) + (0.2 * time_accuracy) + (0.1 * (1.0 - false_alarm_rate))
        
        return metrics, score, target_lead_time
    
    # 尋找最佳閾值
    if find_best_threshold:
        thresholds = np.arange(0.1, 0.7, 0.05)  # 修改閾值範圍，使用更低的閾值
        best_threshold = 0.3  # 默認最佳閾值從0.9降至0.3
        best_score = -1
        best_metrics = None
        target_lead_time = 7  # 目標提前時間（秒）
        
        print("\n尋找最佳閾值...")
        for threshold in thresholds:
            metrics, score, _ = evaluate_with_threshold(threshold)
            print(f"閾值: {threshold:.2f}, 分數: {score:.4f}")
            
            avg_early = np.mean(metrics['early_time_diffs']) if metrics['early_time_diffs'] else 0
            print(f"檢測: {metrics['detections']}, 提前檢測: {metrics['early_detections']}, "
                  f"平均提前時間: {avg_early:.2f}秒, 誤報: {metrics['false_alarms']}")
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_metrics = metrics
        
        print(f"\n最佳閾值: {best_threshold:.2f}, 最佳分數: {best_score:.4f}")
        return best_metrics, best_threshold, target_lead_time
    else:
        # 使用0.3作為閾值
        metrics, score, target_lead_time = evaluate_with_threshold(0.3)
        return metrics, 0.3, target_lead_time

# 在evaluate_predictions函數之後，time_difference_loss函數之前添加這個函數

def find_optimal_threshold(pred_values, window_size=100):
    """
    根據預測值數據自動尋找最佳閾值，專注於識別明顯的方波峰值。
    
    此算法基於以下原理：
    1. 離床事件通常表現為預測值的尖峰(spike)或方波
    2. 方波峰值在前後與背景值有明顯差異
    3. 使用更激進的閾值策略來只捕捉顯著的峰值
    
    參數:
    - pred_values: 模型預測的概率值數組
    - window_size: 局部窗口大小，用於分析數據的局部特性
    
    返回:
    - optimal_threshold: 計算出的最佳閾值
    """
    if len(pred_values) == 0:
        return 0.5  # 默認閾值
    
    # 轉換為numpy數組以進行計算
    if not isinstance(pred_values, np.ndarray):
        pred_values = np.array(pred_values)
    
    # 檢測預測值的統計特性
    pred_mean = np.mean(pred_values)
    pred_std = np.std(pred_values)
    pred_max = np.max(pred_values)
    pred_min = np.min(pred_values)
    
    print(f"預測值統計: 平均={pred_mean:.4f}, 標準差={pred_std:.4f}, 最大值={pred_max:.4f}, 最小值={pred_min:.4f}")
    
    # 方法1: 增強型統計閾值 - 使用更大的系數來確保只檢測顯著峰值
    # 對於方波特性的數據，我們需要更高的閾值來區分峰值與背景
    statistical_threshold = pred_mean + 2.5 * pred_std  # 增大倍數從1.5到2.5
    
    # 方法2: 改進的峰值檢測法 - 聚焦於檢測顯著的尖峰
    peak_thresholds = []
    significant_peaks = []
    
    # 滑動窗口分析，尋找顯著峰值
    for i in range(0, len(pred_values), window_size//2):  # 增加窗口重疊以捕捉更多峰值
        window = pred_values[i:i+window_size]
        if len(window) < 10:  # 確保窗口有足夠的數據
            continue
            
        window_mean = np.mean(window)
        window_std = np.std(window)
        window_max = np.max(window)
        
        # 更嚴格的峰值判定標準 - 尋找與窗口均值有明顯差異的峰值
        if window_max > window_mean + 3.0 * window_std:  # 提高標準差倍數
            # 檢查峰值是否形成方波特徵（有一定持續時間）
            peak_indices = np.where(window > window_mean + 2.0 * window_std)[0]
            if len(peak_indices) > 0:
                # 計算峰值點的平均值作為閾值參考
                peak_mean = np.mean(window[peak_indices])
                significant_peaks.append(peak_mean)
                
                # 計算適合當前窗口的閾值 - 基於窗口均值和峰值的差距
                window_threshold = window_mean + (peak_mean - window_mean) * 0.6  # 設定為峰值和背景之間的值
                peak_thresholds.append(window_threshold)
    
    # 方法3: 高百分位數法 - 使用更高的百分位數來捕捉稀疏的峰值
    # 對於有少量明顯峰值的數據，99百分位比95百分位更合適
    percentile_threshold = np.percentile(pred_values, 99)
    
    # 方法4: 最大梯度法 - 找出預測值分布中的顯著跳變點
    sorted_preds = np.sort(pred_values)
    if len(sorted_preds) > 100:
        # 計算排序後的梯度
        gradients = np.diff(sorted_preds)
        
        # 找出最大梯度點
        max_gradient_idx = np.argmax(gradients)
        
        # 使用最大梯度點作為閾值，或其附近的值
        if max_gradient_idx < len(sorted_preds) - 1:
            gradient_threshold = sorted_preds[max_gradient_idx + 1]
        else:
            gradient_threshold = sorted_preds[-1] * 0.85
            
        # 檢查是否有多個顯著梯度點（說明有多種分布）
        significant_gradients = np.where(gradients > np.mean(gradients) + 2 * np.std(gradients))[0]
        if len(significant_gradients) > 1:
            # 選擇較高的梯度點作為閾值
            high_gradient_idx = significant_gradients[-1]  # 取最後一個顯著梯度點
            if high_gradient_idx < len(sorted_preds) - 1:
                gradient_threshold = sorted_preds[high_gradient_idx + 1]
    else:
        gradient_threshold = 0.5
    
    # 方法5: 直方圖分析法 - 尋找數據分布中的顯著分隔
    hist, bin_edges = np.histogram(pred_values, bins=50)
    # 找出直方圖中的谷值，作為潛在的閾值
    hist_valley_indices = np.where((hist[1:-1] < hist[:-2]) & (hist[1:-1] < hist[2:]))[0] + 1
    
    histogram_threshold = 0.5  # 默認值
    if len(hist_valley_indices) > 0 and np.max(pred_values) > 0.6:
        # 尋找直方圖中位於高值區域的谷點
        high_valleys = [i for i in hist_valley_indices if bin_edges[i] > pred_mean]
        if high_valleys:
            # 選擇高值區域中的第一個谷點作為閾值
            histogram_threshold = bin_edges[high_valleys[0]]
    
    # 綜合多種方法，選擇最合適的閾值
    if len(significant_peaks) > 0:
        # 如果檢測到顯著峰值，使用基於峰值特性的閾值
        peak_mean = np.mean(significant_peaks)
        background_mean = pred_mean
        
        # 閾值設定為峰值和背景均值之間的值，偏向峰值
        peak_based_threshold = background_mean + (peak_mean - background_mean) * 0.6
        
        # 收集所有計算出的閾值
        all_thresholds = [statistical_threshold, percentile_threshold, gradient_threshold, histogram_threshold]
        if len(peak_thresholds) > 0:
            all_thresholds.append(np.mean(peak_thresholds))
        all_thresholds.append(peak_based_threshold)
        
        # 根據數據特性選擇合適的閾值策略
        if pred_max > 0.85 and len(significant_peaks) <= 5:
            # 數據中有少量明顯的峰值，使用較高閾值避免誤報
            optimal_threshold = np.median(all_thresholds)
        else:
            # 根據峰值和背景的對比度選擇閾值
            contrast_ratio = peak_mean / (background_mean + 1e-6)
            if contrast_ratio > 2.0:  # 峰值與背景反差明顯
                # 使用峰值與背景之間的值，偏向峰值
                optimal_threshold = peak_based_threshold
            else:
                # 使用統計方法
                optimal_threshold = np.median(all_thresholds)
    else:
        # 沒有檢測到顯著的峰值，使用保守的閾值策略
        all_thresholds = [statistical_threshold, percentile_threshold, gradient_threshold, histogram_threshold]
        
        # 檢查數據的變異性
        variation_coefficient = pred_std / (pred_mean + 1e-6)
        if variation_coefficient > 0.5:  # 數據變異較大
            # 使用較低閾值以避免漏報
            optimal_threshold = np.min([statistical_threshold, percentile_threshold])
        else:
            # 保守策略，使用較高閾值以避免誤報
            optimal_threshold = np.max([statistical_threshold, percentile_threshold])
    
    # 對於稀疏峰值的情況，進一步提高閾值
    if np.sum(pred_values > percentile_threshold) < len(pred_values) * 0.01:
        # 使用99.5百分位作為最低閾值
        min_threshold = np.percentile(pred_values, 99.5)
        optimal_threshold = max(optimal_threshold, min_threshold)
    
    # 確保閾值在有意義的範圍內，但提高最小閾值以更加聚焦於顯著峰值
    optimal_threshold = max(0.4, min(optimal_threshold, 0.9))
    
    print(f"計算的最佳閾值: {optimal_threshold:.4f}")
    print(f"統計閾值: {statistical_threshold:.4f}, 百分位閾值: {percentile_threshold:.4f}, 梯度閾值: {gradient_threshold:.4f}")
    if len(peak_thresholds) > 0:
        print(f"峰值閾值: {np.mean(peak_thresholds):.4f} (從 {len(peak_thresholds)} 個窗口計算)")
    
    return optimal_threshold

# 自定義的時間差異 loss function
def time_difference_loss(y_true, y_pred):
    """
    純粹基於時間差異的損失函數
    
    目標：
    1. 鼓勵模型在實際離床事件前進行預測
    2. 對於提前預測，時間越接近實際離床時間越好（但不要晚於實際時間）
    3. 嚴懲晚於實際離床時間的預測
    
    注意：這個函數假設在批次訓練中，序列是按時間順序排列的
    """
    # 轉換為張量
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # 找出實際的離床事件位置（標籤為1的地方）
    event_mask = tf.cast(tf.equal(y_true, 1.0), tf.float32)
    
    # 根據預測值和閾值創建預測掩碼
    pred_mask = tf.cast(tf.greater_equal(y_pred, 0.5), tf.float32)
    
    # 初始化損失
    time_loss = tf.zeros_like(y_true)
    
    # 計算預測損失 - 根據預測與實際事件的時間關係
    
    # 1. 成功預測損失（True Positive）- 希望預測接近但提前於實際事件
    # 對於每個預測為1且標籤為1的位置，損失 = 0（完美預測）
    exact_match = tf.multiply(pred_mask, event_mask)
    time_loss = tf.where(
        tf.greater(exact_match, 0),
        tf.zeros_like(time_loss),  # 完美預測，損失為0
        time_loss
    )
    
    # 2. 真實事件前的預測（提前預測）- 輕微懲罰，但比晚預測好
    # 注意：這裡假設批次中的序列是時間連續的
    # 這部分計算會很複雜，因為它涉及到時間關係
    
    # 3. 嚴懲漏報 - 標籤為1但預測為0
    # 對於每個標籤為1但預測為0的位置，給予嚴重懲罰
    missed_events = tf.multiply(tf.subtract(1.0, pred_mask), event_mask)
    time_loss = tf.where(
        tf.greater(missed_events, 0),
        tf.ones_like(time_loss) * 10.0,  # 嚴重懲罰漏報
        time_loss
    )
    
    # 4. 嚴懲誤報 - 標籤為0但預測為1
    # 對於每個標籤為0但預測為1的位置，給予中等懲罰
    false_alarms = tf.multiply(pred_mask, tf.subtract(1.0, event_mask))
    time_loss = tf.where(
        tf.greater(false_alarms, 0),
        tf.ones_like(time_loss) * 5.0,  # 中等懲罰誤報
        time_loss
    )
    
    # 5. 正確的負例 - 不需懲罰
    true_negatives = tf.multiply(tf.subtract(1.0, pred_mask), tf.subtract(1.0, event_mask))
    time_loss = tf.where(
        tf.greater(true_negatives, 0),
        tf.zeros_like(time_loss),  # 正確的負例，損失為0
        time_loss
    )
    
    return tf.reduce_mean(time_loss)

# 自定義的時間差差異 loss function（更精準的實現）
def sequence_time_difference_loss(y_true, y_pred):
    """
    考慮序列中的時間關係，計算時間差異損失
    
    這個損失函數鼓勵模型：
    1. 在實際離床事件發生前進行預測（但不要太早）
    2. 懲罰晚於實際事件的預測，懲罰程度與延遲時間成正比
    """
    # 在TensorFlow中實現時間差異損失比較複雜，這裡使用一個簡化的方法
    # 我們假設在每個批次中，序列是按時間順序排列的
    
    # 轉換為二值預測
    threshold = 0.9  # 調整閾值為0.9
    y_pred_binary = tf.cast(tf.greater_equal(y_pred, threshold), tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    
    # 基礎損失 - 二元交叉熵
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    
    # 對離床事件的漏報進行嚴重懲罰（標籤為1但預測為0）
    false_negative_penalty = tf.where(
        tf.logical_and(tf.less(y_pred, threshold), tf.equal(y_true, 1.0)),
        tf.ones_like(y_true) * 10.0,  # 嚴重懲罰漏報
        tf.ones_like(y_true)
    )
    
    # 對非離床時間的誤報進行中等懲罰（標籤為0但預測為1）
    false_positive_penalty = tf.where(
        tf.logical_and(tf.greater_equal(y_pred, threshold), tf.equal(y_true, 0.0)),
        tf.ones_like(y_true) * 5.0,  # 中等懲罰誤報
        tf.ones_like(y_true)
    )
    
    # 時間敏感懲罰
    # 提前預測被視為好的結果，受到獎勵
    early_prediction_reward = tf.where(
        tf.logical_and(tf.greater_equal(y_pred, threshold), tf.equal(y_true, 1.0)),
        tf.ones_like(y_true) * 0.5,  # 獎勵提前預測（減輕損失）
        tf.ones_like(y_true)
    )
    
    # 計算總體損失
    weighted_loss = bce * false_negative_penalty * false_positive_penalty * early_prediction_reward
    
    return tf.reduce_mean(weighted_loss)

# 修改pure_time_difference_loss函數
def pure_time_difference_loss(y_true, y_pred, target_lead_time=7.0):
    """
    專為離床預測設計的時間差異損失函數
    
    參數:
    - y_true: 實際標籤（從0到1的漸進式標籤）
    - y_pred: 預測概率
    - target_lead_time: 目標提前預測時間（默認7秒）
    """
    # 轉換為張量
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # 基礎損失 - 均方誤差，適合漸進式標籤
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # 區分不同預測情況，定義3種區域遮罩
    # 1. 關鍵離床區域（標籤>=0.7)
    critical_event_mask = tf.cast(tf.greater_equal(y_true, 0.7), tf.float32)
    
    # 2. 預警區域 (0.5 <= 標籤 < 0.7)
    warning_mask = tf.cast(
        tf.logical_and(
            tf.greater_equal(y_true, 0.5),
            tf.less(y_true, 0.7)
        ), 
        tf.float32
    )
    
    # 3. 非事件區域 (標籤<0.5)
    normal_mask = tf.cast(tf.less(y_true, 0.5), tf.float32)
    
    # 使用較低的預測閾值，增加模型敏感度
    prediction_threshold = 0.5
    
    # 使用上面定義的遮罩直接計算不同類型的錯誤
    # 關鍵離床事件漏報懲罰 (最嚴重錯誤) - 使用critical_event_mask
    critical_miss_mask = tf.logical_and(
        tf.less(y_pred, prediction_threshold),
        tf.equal(critical_event_mask, 1.0)
    )
    critical_miss_penalty = tf.where(
        critical_miss_mask,
        tf.ones_like(y_true) * 25.0,  # 提高懲罰權重
        tf.ones_like(y_true)
    )
    
    # 預警區漏報懲罰 - 使用warning_mask
    warning_miss_mask = tf.logical_and(
        tf.less(y_pred, prediction_threshold),
        tf.equal(warning_mask, 1.0)
    )
    warning_miss_penalty = tf.where(
        warning_miss_mask,
        tf.ones_like(y_true) * 10.0,  # 提高預警區重要性
        tf.ones_like(y_true)
    )
    
    # 誤報懲罰 (較嚴重錯誤，但不如漏報) - 使用normal_mask
    false_alarm_mask = tf.logical_and(
        tf.greater_equal(y_pred, prediction_threshold),
        tf.less(y_true, 0.3)  # 更嚴格的誤報定義
    )
    false_alarm_penalty = tf.where(
        false_alarm_mask,
        tf.ones_like(y_true) * 8.0,
        tf.ones_like(y_true)
    )
    
    # 輕度誤報懲罰 - 使用normal_mask
    minor_false_alarm_mask = tf.logical_and(
        tf.logical_and(
            tf.greater_equal(y_pred, 0.3),
            tf.less(y_pred, prediction_threshold)
        ),
        tf.less(y_true, 0.3)
    )
    minor_false_alarm_penalty = tf.where(
        minor_false_alarm_mask,
        tf.ones_like(y_true) * 2.0,
        tf.ones_like(y_true)
    )
    
    # 成功預測獎勵 - 使用 critical_event_mask 和 warning_mask 的組合
    success_mask = tf.logical_and(
        tf.greater_equal(y_pred, prediction_threshold),
        tf.logical_or(
            tf.equal(critical_event_mask, 1.0),
            tf.equal(warning_mask, 1.0)
        )
    )
    success_reward = tf.where(
        success_mask,
        tf.ones_like(y_true) * 0.1,  # 更大的獎勵（通過更小的乘數）
        tf.ones_like(y_true)
    )
    
    # 使用加法而非乘法來組合懲罰因子，避免數值問題
    # 計算各種罰分的總權重
    penalty_weights = (
        15.0 * tf.reduce_sum(tf.cast(critical_miss_mask, tf.float32)) +
        5.0 * tf.reduce_sum(tf.cast(warning_miss_mask, tf.float32)) +
        3.0 * tf.reduce_sum(tf.cast(false_alarm_mask, tf.float32)) +
        1.0 * tf.reduce_sum(tf.cast(minor_false_alarm_mask, tf.float32)) -
        2.0 * tf.reduce_sum(tf.cast(success_mask, tf.float32))  # 獎勵使用負值
    )
    
    # 添加一個小常數避免除零錯誤
    batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
    normalized_penalty = penalty_weights / (batch_size + 1e-7)
    
    # 最終損失 = 原始損失 + 懲罰項
    weighted_loss = mse + 0.3 * normalized_penalty
    
    return weighted_loss

def improved_time_difference_loss(y_true, y_pred, target_lead_time=7.0):
    """
    改進版時間差異損失函數
    
    參數:
    - y_true: 實際標籤
    - y_pred: 預測概率
    - target_lead_time: 目標提前預測時間（秒）
    """
    # 轉換為張量
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # 結合BCE和MSE的基礎損失
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    base_loss = 0.7 * bce + 0.3 * mse
    
    # 定義預測閾值
    prediction_threshold = 0.7
    
    # 區分不同區域遮罩
    critical_event_mask = tf.cast(tf.greater_equal(y_true, 0.7), tf.float32)
    warning_mask = tf.cast(
        tf.logical_and(
            tf.greater_equal(y_true, 0.5),
            tf.less(y_true, 0.7)
        ), 
        tf.float32
    )
    normal_mask = tf.cast(tf.less(y_true, 0.5), tf.float32)
    
    # 使用遮罩創建不同類型的錯誤遮罩
    # 1. 關鍵離床事件漏報
    critical_miss_mask = tf.logical_and(
        tf.less(y_pred, prediction_threshold),
        tf.equal(critical_event_mask, 1.0)
    )
    critical_miss = tf.cast(critical_miss_mask, tf.float32) * 10.0
    
    # 2. 預警區漏報
    warning_miss_mask = tf.logical_and(
        tf.less(y_pred, prediction_threshold),
        tf.equal(warning_mask, 1.0)
    )
    warning_miss = tf.cast(warning_miss_mask, tf.float32) * 5.0
    
    # 3. 嚴重誤報 - 高機率預測但實際是正常狀態
    false_alarm_mask = tf.logical_and(
        tf.greater_equal(y_pred, prediction_threshold),
        tf.less(y_true, 0.3)
    )
    false_alarm = tf.cast(false_alarm_mask, tf.float32) * 3.0
    
    # 4. 輕度誤報 - 中等機率預測但實際是正常狀態
    minor_false_alarm_mask = tf.logical_and(
        tf.logical_and(
            tf.greater_equal(y_pred, 0.5),
            tf.less(y_pred, prediction_threshold)
        ),
        tf.less(y_true, 0.3)
    )
    minor_false_alarm = tf.cast(minor_false_alarm_mask, tf.float32) * 1.0
    
    # 5. 成功預測獎勵 - 使用critical_event_mask和warning_mask
    success_mask = tf.logical_and(
        tf.greater_equal(y_pred, prediction_threshold),
        tf.logical_or(
            tf.equal(critical_event_mask, 1.0),
            tf.equal(warning_mask, 1.0)
        )
    )
    success_reward = tf.cast(success_mask, tf.float32) * (-1.0)  # 負值表示獎勵
    
    # 組合所有懲罰和獎勵（使用加法而非乘法）
    total_penalty = critical_miss + warning_miss + false_alarm + minor_false_alarm + success_reward
    
    # 總體損失 = 基礎損失 + 加權懲罰
    final_loss = base_loss + 0.5 * total_penalty
    
    return tf.reduce_mean(final_loss)

def simplified_time_difference_loss(y_true, y_pred, target_lead_time=7.0):
    """
    簡化版時間差異損失函數 - 針對二進制標籤資料優化
    調整參數以提高預測機率
    
    參數:
    - y_true: 實際標籤 (二進制 0/1)
    - y_pred: 預測概率
    - target_lead_time: 目標提前預測時間（默認7秒）
    """
    # 轉換為張量
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # 基礎損失 - 結合BCE和MSE，增加BCE權重
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    base_loss = 0.7 * bce + 0.3 * mse  # 適當減少BCE權重，增加MSE權重
    
    # 二進制遮罩 - 只區分事件和非事件
    event_mask = tf.cast(tf.equal(y_true, 1.0), tf.float32)  # 離床事件
    non_event_mask = tf.cast(tf.equal(y_true, 0.0), tf.float32)  # 非離床事件
    
    # 預測閾值 - 降低各閾值以增加預測機率
    prediction_threshold = 0.4  # 主閾值從0.5降到0.4
    high_confidence = 0.7      # 高置信度閾值從0.8降到0.7
    low_confidence = 0.2       # 低置信度閾值從0.3降到0.2
    
    # 1. 漏報懲罰 (最嚴重的錯誤) - 大幅提高懲罰力度
    miss_mask = tf.logical_and(
        tf.less(y_pred, prediction_threshold),
        tf.equal(y_true, 1.0)
    )
    # 漏報程度越嚴重，懲罰越高 (預測值越低懲罰越高)
    miss_degree = 1.0 - y_pred
    miss_penalty = tf.where(
        miss_mask, 
        25.0 * miss_degree * event_mask,  # 提高懲罰係數從15.0到25.0
        tf.zeros_like(y_pred)
    )
    
    # 2. 高置信度誤報懲罰 (較嚴重的錯誤) - 適當降低懲罰力度
    severe_false_alarm_mask = tf.logical_and(
        tf.greater_equal(y_pred, high_confidence),
        tf.equal(y_true, 0.0)
    )
    severe_false_alarm_penalty = tf.where(
        severe_false_alarm_mask,
        9.0 * y_pred * non_event_mask,  # 降低懲罰係數從12.0到9.0
        tf.zeros_like(y_pred)
    )
    
    # 3. 中度誤報懲罰 - 降低懲罰力度
    medium_false_alarm_mask = tf.logical_and(
        tf.logical_and(
            tf.greater_equal(y_pred, prediction_threshold),
            tf.less(y_pred, high_confidence)
        ),
        tf.equal(y_true, 0.0)
    )
    medium_false_alarm_penalty = tf.where(
        medium_false_alarm_mask,
        4.0 * y_pred * non_event_mask,  # 降低懲罰係數從6.0到4.0
        tf.zeros_like(y_pred)
    )
    
    # 4. 輕度誤報懲罰 - 降低懲罰力度
    mild_false_alarm_mask = tf.logical_and(
        tf.logical_and(
            tf.greater_equal(y_pred, low_confidence),
            tf.less(y_pred, prediction_threshold)
        ),
        tf.equal(y_true, 0.0)
    )
    mild_false_alarm_penalty = tf.where(
        mild_false_alarm_mask,
        1.0 * y_pred * non_event_mask,  # 降低懲罰係數從2.0到1.0
        tf.zeros_like(y_pred)
    )
    
    # 5. 成功預測獎勵 - 大幅增加獎勵力度
    success_mask = tf.logical_and(
        tf.greater_equal(y_pred, prediction_threshold),
        tf.equal(y_true, 1.0)
    )
    # 預測置信度越高獎勵越大
    success_reward = tf.where(
        success_mask,
        -6.0 * y_pred * event_mask,  # 增加獎勵係數從-3.0到-6.0
        tf.zeros_like(y_pred)
    )
    
    # 6. 平滑性懲罰 - 降低懲罰力度
    y_pred_shifted = tf.roll(y_pred, shift=1, axis=0)
    prediction_jumps = tf.abs(y_pred - y_pred_shifted)
    
    # 針對兩種情況的跳變做懲罰處理:
    # 1. 非事件區域中的大幅跳變 (減少噪聲) - 提高跳變閾值，降低懲罰
    non_event_jumps_mask = tf.logical_and(
        tf.greater(prediction_jumps, 0.3),  # 提高跳變閾值從0.2到0.3
        tf.equal(y_true, 0.0)
    )
    
    # 2. 事件區域中的過於尖銳的峰值 (讓預測更平滑) - 提高跳變閾值
    event_jumps_mask = tf.logical_and(
        tf.greater(prediction_jumps, 0.5),  # 提高跳變閾值從0.4到0.5
        tf.equal(y_true, 1.0)
    )
    
    # 綜合平滑處理 - 降低懲罰力度
    smoothness_penalty = tf.where(
        non_event_jumps_mask,
        2.0 * prediction_jumps * non_event_mask,  # 降低懲罰係數從3.0到2.0
        tf.zeros_like(y_pred)
    ) + tf.where(
        event_jumps_mask,
        1.0 * prediction_jumps * event_mask,  # 降低懲罰係數從1.5到1.0
        tf.zeros_like(y_pred)
    )
    
    # 組合所有懲罰和獎勵
    total_penalty = (
        miss_penalty + 
        severe_false_alarm_penalty + 
        medium_false_alarm_penalty +
        mild_false_alarm_penalty + 
        success_reward +  
        smoothness_penalty
    )
    
    # 計算最終損失 - 增加自定義懲罰和獎勵的權重
    final_loss = base_loss + 0.9 * tf.reduce_mean(total_penalty)  # 提高懲罰權重從0.7到0.9
    
    return final_loss

def build_model(input_shape, loss_type='focal'):
    """
    構建用於處理完整時間序列的模型
    
    參數:
    - input_shape: 輸入數據的形狀，例如 (time_steps, features)
    - loss_type: 使用的損失函數類型，可選 'focal'(默認)、'time_diff'、'improved_time_diff'、'simplified_time_diff'
    
    返回:
    - 構建好的模型
    """
    input_layer = Input(shape=input_shape)
    
    # CNN層用於特徵提取
    conv1 = Conv1D(128, kernel_size=5, padding='same', activation='relu')(input_layer)  # 增加神經元
    conv1 = BatchNormalization()(conv1)
    
    # 二級CNN層
    conv2 = Conv1D(128, kernel_size=5, padding='same', activation='relu')(conv1)  # 增加神經元
    conv2 = BatchNormalization()(conv2)
    conv2 = Add()([conv1, conv2])
    
    # 使用MaxPooling減少序列長度，提高計算效率
    # 檢查輸入維度，確保池化大小不會太大
    # 根據輸入形狀動態調整池化大小
    pool_size = min(4, input_shape[0] // 2)  # 確保池化大小不超過序列長度的一半
    if pool_size > 0:
        x = MaxPooling1D(pool_size=pool_size)(conv2)
    else:
        x = conv2  # 如果無法池化，直接使用卷積輸出
    x = Dropout(0.25)(x)
    
    # 多層雙向LSTM用於捕獲時間依賴關係
    x = Bidirectional(LSTM(256, return_sequences=True))(x)  # 增加神經元
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    
    # 第二層LSTM
    x = Bidirectional(LSTM(128, return_sequences=True))(x)  # 增加神經元
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    
    # 第三層LSTM
    x = Bidirectional(LSTM(64, return_sequences=False))(x)  # 設置為False，不返回序列，增加神經元
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    
    # 使用Dense層將特徵壓縮到單一輸出
    x = Dense(32, activation='relu')(x)  # 增加神經元
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    
    # 輸出層
    output_layer = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    # 定義Focal Loss以更好地處理不平衡問題
    def focal_loss(gamma=2.0, alpha=0.25):
        def focal_loss_fixed(y_true, y_pred):
            pt = tf.where(tf.equal(y_true, 1), y_pred, 1-y_pred)
            alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1-alpha)
            return -alpha_t * (1-pt)**gamma * tf.math.log(pt + 1e-7)  # 添加一個小值避免log(0)
        return focal_loss_fixed
    
    # 根據選擇的損失函數類型進行編譯
    if loss_type == 'time_diff':
        # 使用pure_time_difference_loss
        print("使用時間差異損失函數 (pure_time_difference_loss)")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=pure_time_difference_loss,  # 使用時間差異損失
            metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), 
                    tf.keras.metrics.AUC()]
        )
    elif loss_type == 'improved_time_diff':
        # 使用improved_time_difference_loss
        print("使用改進版時間差異損失函數 (improved_time_difference_loss)")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=improved_time_difference_loss,  # 使用改進版時間差異損失
            metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), 
                    tf.keras.metrics.AUC()]
        )
    elif loss_type == 'simplified_time_diff':
        # 使用simplified_time_difference_loss
        print("使用簡化版時間差異損失函數 (simplified_time_difference_loss)")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=simplified_time_difference_loss,  # 使用簡化版時間差異損失
            metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), 
                    tf.keras.metrics.AUC()]
        )
    else:
        # 使用Focal Loss (默認)
        print("使用Focal Loss")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=focal_loss(gamma=2.0, alpha=0.85),  # 增加alpha值，更偏重少數類別
            metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), 
                    tf.keras.metrics.AUC()]
        )
    
    return model

def plot_event_timeline(y_true, y_pred, timestamps, threshold=0.9):
    """
    繪製時間軸上的事件對比圖，顯示實際事件與預測事件
    
    參數:
    - y_true: 實際標籤
    - y_pred: 預測概率
    - timestamps: 時間戳列表
    - threshold: 預測閾值，用於轉換概率為二進制預測
    """
    # 創建二值預測
    predictions = (y_pred >= threshold).astype(int)
    
    # 找出所有預測事件 - 連續的預測為1的段落被視為一個事件
    predicted_events = []
    i = 0
    while i < len(predictions):
        if predictions[i] == 1:
            start_idx = i
            while i < len(predictions) and predictions[i] == 1:
                i += 1
            end_idx = i - 1
            
            # 儲存事件開始和結束時間
            predicted_events.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_time': timestamps[start_idx],
                'end_time': timestamps[end_idx]
            })
        else:
            i += 1
    
    # 找出所有實際事件 - 標籤超過閾值（對於漸進式標籤，通常是0.7）被視為離床事件
    actual_events = []
    i = 0
    high_threshold = 0.7  # 用於識別實際離床事件的閾值
    while i < len(y_true):
        if y_true[i] >= high_threshold:
            start_idx = i
            # 找出這個事件的最高標籤點（事件發生時刻）
            max_label_idx = start_idx
            max_label = y_true[start_idx]
            
            while i < len(y_true) and y_true[i] >= 0.5:  # 尋找連續的事件區域
                if y_true[i] > max_label:
                    max_label = y_true[i]
                    max_label_idx = i
                i += 1
            end_idx = i - 1
            
            # 儲存事件信息
            actual_events.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'peak_idx': max_label_idx,  # 事件最高點（通常是離床時刻）
                'start_time': timestamps[start_idx],
                'end_time': timestamps[end_idx],
                'peak_time': timestamps[max_label_idx]
            })
        else:
            i += 1
    
    # 繪製時間軸圖
    plt.figure(figsize=(20, 6))
    
    # 設定X軸範圍
    min_time = min(timestamps)
    max_time = max(timestamps)
    plt.xlim(min_time, max_time)
    
    # 繪製標籤值曲線（灰色背景參考）
    plt.plot(timestamps, y_true, 'gray', alpha=0.3, label='真實標籤值')
    
    # 繪製預測值曲線（灰色背景參考）
    plt.plot(timestamps, y_pred, 'lightblue', alpha=0.3, label='預測概率值')
    
    # 繪製實際事件（紅色區塊）
    for event in actual_events:
        # 繪製事件區間
        plt.axvspan(event['start_time'], event['end_time'], alpha=0.3, color='red')
        # 繪製事件峰值點（實際離床時刻）
        plt.axvline(x=event['peak_time'], color='darkred', linestyle='-', linewidth=2)
    
    # 繪製預測事件（綠色區塊）
    for event in predicted_events:
        plt.axvspan(event['start_time'], event['end_time'], alpha=0.3, color='green')
    
    # 添加圖例和標籤
    plt.hlines(y=threshold, xmin=min_time, xmax=max_time, 
              colors='blue', linestyles='dashed', label=f'預測閾值 ({threshold})')
    plt.hlines(y=high_threshold, xmin=min_time, xmax=max_time, 
              colors='red', linestyles='dashed', label=f'實際事件閾值 ({high_threshold})')
    
    # 設定圖表標題和軸標籤
    plt.title('離床事件與AI預測的時間對比圖')
    plt.xlabel('時間點')
    plt.ylabel('事件概率值')
    plt.legend(loc='upper right')
    
    # 添加說明文字
    plt.figtext(0.5, 0.01, 
                '紅色區塊: 實際離床事件區間 | 深紅色線: 實際離床時刻 | 綠色區塊: AI預測事件區間', 
                ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    # 保存圖表
    plt.tight_layout()
    event_timeline_path = os.path.join(LOG_DIR, 'event_timeline_comparison.png')
    plt.savefig(event_timeline_path)
    print(f"事件時間軸對比圖已保存至: {event_timeline_path}")
    plt.close()

def plot_event_time_differences(y_true, y_pred, timestamps, threshold=0.9):
    """
    繪製詳細的事件時間差異對比圖，展示每個事件的預測時間與實際時間的差異
    
    參數:
    - y_true: 實際標籤
    - y_pred: 預測概率
    - timestamps: 時間戳列表
    - threshold: 預測閾值
    """
    # 創建二值預測
    predictions = (y_pred >= threshold).astype(int)
    
    # 找出所有預測事件
    predicted_events = []
    i = 0
    while i < len(predictions):
        if predictions[i] == 1:
            start_idx = i
            while i < len(predictions) and predictions[i] == 1:
                i += 1
            end_idx = i - 1
            
            predicted_events.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_time': timestamps[start_idx],
                'end_time': timestamps[end_idx]
            })
        else:
            i += 1
    
    # 找出所有實際事件
    actual_events = []
    i = 0
    high_threshold = 0.7
    while i < len(y_true):
        if y_true[i] >= high_threshold:
            start_idx = i
            max_label_idx = start_idx
            max_label = y_true[start_idx]
            
            while i < len(y_true) and y_true[i] >= 0.5:
                if y_true[i] > max_label:
                    max_label = y_true[i]
                    max_label_idx = i
                i += 1
            end_idx = i - 1
            
            actual_events.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'peak_idx': max_label_idx,
                'start_time': timestamps[start_idx],
                'end_time': timestamps[end_idx],
                'peak_time': timestamps[max_label_idx]
            })
        else:
            i += 1
    
    # 匹配實際事件與預測事件
    matched_events = []
    for actual_event in actual_events:
        best_pred_event = None
        best_time_diff = float('inf')
        
        for pred_event in predicted_events:
            # 計算預測開始時間與實際事件發生時間的差距
            time_diff = pred_event['start_time'] - actual_event['peak_time']
            
            # 我們認為以下情況是有效的檢測
            if -WARNING_TIME <= time_diff <= 5:
                if abs(time_diff) < abs(best_time_diff):
                    best_time_diff = time_diff
                    best_pred_event = pred_event
        
        if best_pred_event is not None:
            matched_events.append({
                'actual_event': actual_event,
                'pred_event': best_pred_event,
                'time_diff': best_time_diff
            })
            # 標記已匹配
            best_pred_event['matched'] = True
        else:
            # 漏報
            matched_events.append({
                'actual_event': actual_event,
                'pred_event': None,
                'time_diff': None
            })
    
    # 找出誤報（未匹配的預測事件）
    false_alarms = [e for e in predicted_events if not e.get('matched', False)]
    
    # 繪製詳細的事件時間差異圖
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # 設置Y軸為事件序號
    event_ids = range(1, len(matched_events) + 1)
    
    # 繪製實際事件時間點（藍色點）
    actual_times = [event['actual_event']['peak_time'] for event in matched_events]
    ax.scatter(actual_times, event_ids, color='blue', s=100, label='實際離床時間')
    
    # 分別繪製提前預測和延遲預測
    early_diffs = []
    early_ids = []
    late_diffs = []
    late_ids = []
    missed_ids = []
    
    for i, match in enumerate(matched_events, 1):
        if match['pred_event'] is not None:
            pred_time = match['pred_event']['start_time']
            # 繪製預測時間點與實際時間點之間的連線
            ax.plot([actual_times[i-1], pred_time], [i, i], 'gray', linewidth=1.5, linestyle='--')
            
            # 分類為提前或延遲預測
            if match['time_diff'] <= 0:  # 提前預測
                early_diffs.append(pred_time)
                early_ids.append(i)
            else:  # 延遲預測
                late_diffs.append(pred_time)
                late_ids.append(i)
        else:
            missed_ids.append(i)
    
    # 繪製提前預測、延遲預測和漏報
    if early_diffs:
        ax.scatter(early_diffs, early_ids, color='green', s=100, label='提前預測')
    if late_diffs:
        ax.scatter(late_diffs, late_ids, color='orange', s=100, label='延遲預測')
    if missed_ids:
        ax.scatter([actual_times[i-1] for i in missed_ids], missed_ids, marker='x', color='red', s=100, label='漏報')
    
    # 繪製誤報
    if false_alarms:
        false_alarm_times = [event['start_time'] for event in false_alarms]
        # 使用較低的Y值來顯示誤報
        false_alarm_y = [0.5] * len(false_alarms)
        ax.scatter(false_alarm_times, false_alarm_y, marker='^', color='red', s=100, label='誤報')
    
    # 添加時間差異標籤
    for i, match in enumerate(matched_events, 1):
        if match['time_diff'] is not None:
            diff_text = f"{abs(match['time_diff']):.1f}秒"
            if match['time_diff'] <= 0:
                diff_text += "(提前)"
                text_color = 'green'
            else:
                diff_text += "(延遲)"
                text_color = 'orange'
            
            # 計算標籤位置
            actual_time = actual_times[i-1]
            pred_time = match['pred_event']['start_time']
            text_x = (actual_time + pred_time) / 2
            
            ax.text(text_x, i+0.2, diff_text, fontsize=8, color=text_color,
                    ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.7))
    
    # 設置圖表標題和軸標籤
    ax.set_title('離床事件預測時間差異對比')
    ax.set_xlabel('時間點')
    ax.set_ylabel('事件編號')
    
    # 添加網格線
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 調整Y軸範圍
    ax.set_ylim(0, len(matched_events) + 1)
    
    # 添加圖例
    ax.legend(loc='upper right')
    
    # 添加說明文字
    plt.figtext(0.5, 0.01, 
                '藍點: 實際離床時間 | 綠點: 提前預測 | 橙點: 延遲預測 | 紅X: 漏報 | 紅三角: 誤報', 
                ha='center', fontsize=11, bbox=dict(facecolor='white', alpha=0.8))
    
    # 添加統計信息
    early_count = len(early_ids)
    late_count = len(late_ids)
    missed_count = len(missed_ids)
    false_count = len(false_alarms)
    
    avg_early_time = np.mean([abs(matched_events[i-1]['time_diff']) for i in early_ids]) if early_ids else 0
    avg_late_time = np.mean([matched_events[i-1]['time_diff'] for i in late_ids]) if late_ids else 0
    
    stats_text = f"統計資訊:\n"
    stats_text += f"總事件數: {len(matched_events)}\n"
    stats_text += f"提前預測: {early_count} (平均提前 {avg_early_time:.2f}秒)\n"
    stats_text += f"延遲預測: {late_count} (平均延遲 {avg_late_time:.2f}秒)\n"
    
    if early_diffs:
        target_diffs = [abs(abs(diff) - target_lead_time) for diff in early_diffs]
        accurate_count = sum(1 for diff in target_diffs if diff <= 3)
        accuracy_rate = (accurate_count / len(target_diffs)) * 100 if target_diffs else 0
        stats_text += f"目標時間準確率: {accuracy_rate:.1f}% (±3秒內)"
    
    plt.figtext(0.5, 0.01, stats_text, fontsize=10, ha='center', 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # 保存圖表
    plt.tight_layout()
    time_diff_path = os.path.join(LOG_DIR, 'event_time_differences.png')
    plt.savefig(time_diff_path)
    print(f"事件時間差異對比圖已保存至: {time_diff_path}")
    plt.close()

def plot_prediction_accuracy_trend(y_true, y_pred, timestamps, threshold=0.9, window_size=50):
    """
    繪製預測準確度隨時間變化的趨勢圖
    
    參數:
    - y_true: 實際標籤
    - y_pred: 預測概率
    - timestamps: 時間戳列表
    - threshold: 預測閾值
    - window_size: 滑動窗口大小，用於計算移動平均準確度
    """
    # 創建二值預測
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # 計算逐點準確度 (0表示錯誤預測，1表示正確預測)
    point_accuracy = np.zeros(len(y_true))
    
    # 定義正確預測的條件:
    # 1. 真正例：實際標籤>=0.7且預測為1
    # 2. 真負例：實際標籤<0.5且預測為0
    # 對於預警區域(0.5<=標籤<0.7)，我們視為中間狀態，若預測為1也算正確
    for i in range(len(y_true)):
        if (y_true[i] >= 0.7 and y_pred_binary[i] == 1) or \
           (0.5 <= y_true[i] < 0.7 and y_pred_binary[i] == 1) or \
           (y_true[i] < 0.5 and y_pred_binary[i] == 0):
            point_accuracy[i] = 1
    
    # 計算累積準確度
    cumulative_accuracy = np.cumsum(point_accuracy) / np.arange(1, len(point_accuracy) + 1)
    
    # 計算滑動窗口準確度
    moving_accuracy = np.zeros(len(point_accuracy))
    for i in range(len(point_accuracy)):
        start_idx = max(0, i - window_size + 1)
        window_points = point_accuracy[start_idx:i+1]
        moving_accuracy[i] = np.mean(window_points)
    
    # 繪製準確度趨勢圖
    plt.figure(figsize=(15, 8))
    
    # 繪製原始標籤和預測值作為參考背景
    plt.plot(timestamps, y_true, 'gray', alpha=0.2, label='真實標籤')
    plt.plot(timestamps, y_pred, 'lightblue', alpha=0.2, label='預測概率')
    
    # 繪製累積和滑動準確度
    plt.plot(timestamps, cumulative_accuracy, 'blue', linewidth=2, label=f'累積準確度')
    plt.plot(timestamps, moving_accuracy, 'red', linewidth=2, label=f'滑動窗口準確度 (窗口大小={window_size})')
    
    # 標記實際離床事件發生的時間點
    event_times = []
    for i in range(1, len(y_true)):
        if y_true[i] >= 0.9 and y_true[i-1] < 0.9:  # 離床事件閾值設為0.9
            event_times.append(timestamps[i])
    
    for t in event_times:
        plt.axvline(x=t, color='red', linestyle='--', alpha=0.5)
    
    # 添加預測閾值參考線
    plt.axhline(y=threshold, color='green', linestyle='--', alpha=0.5, label=f'預測閾值 ({threshold})')
    
    # 設置圖表標題和軸標籤
    plt.title('預測準確度隨時間變化趨勢')
    plt.xlabel('時間點')
    plt.ylabel('準確度 / 預測值')
    
    # 設置Y軸範圍
    plt.ylim(-0.05, 1.05)
    
    # 添加網格線
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 添加圖例
    plt.legend(loc='lower right')
    
    # 添加說明文字
    plt.figtext(0.5, 0.01, 
                '藍線: 累積準確度 | 紅線: 滑動窗口準確度 | 紅色虛線: 離床事件發生時間', 
                ha='center', fontsize=11, bbox=dict(facecolor='white', alpha=0.8))
    
    # 添加窗口大小選擇器的說明
    plt.figtext(0.02, 0.97, 
                f"窗口大小: {window_size}\n"
                f"累積準確度: {cumulative_accuracy[-1]:.4f}\n"
                f"最終窗口準確度: {moving_accuracy[-1]:.4f}", 
                fontsize=10, va='top', 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # 保存圖表
    plt.tight_layout()
    accuracy_trend_path = os.path.join(LOG_DIR, f'prediction_accuracy_trend_w{window_size}.png')
    plt.savefig(accuracy_trend_path)
    print(f"預測準確度趨勢圖已保存至: {accuracy_trend_path}")
    plt.close()

def plot_prediction_time_diff_distribution(y_true, y_pred, timestamps, threshold=0.9):
    """
    繪製預測時間差異的分布圖，包括直方圖和箱型圖
    
    參數:
    - y_true: 實際標籤
    - y_pred: 預測概率
    - timestamps: 時間戳列表
    - threshold: 預測閾值
    """
    # 創建二值預測
    predictions = (y_pred >= threshold).astype(int)
    
    # 找出所有預測事件
    predicted_events = []
    i = 0
    while i < len(predictions):
        if predictions[i] == 1:
            start_idx = i
            while i < len(predictions) and predictions[i] == 1:
                i += 1
            end_idx = i - 1
            
            predicted_events.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_time': timestamps[start_idx],
                'end_time': timestamps[end_idx]
            })
        else:
            i += 1
    
    # 找出所有實際事件
    actual_events = []
    i = 0
    high_threshold = 0.7
    while i < len(y_true):
        if y_true[i] >= high_threshold:
            start_idx = i
            max_label_idx = start_idx
            max_label = y_true[start_idx]
            
            while i < len(y_true) and y_true[i] >= 0.5:
                if y_true[i] > max_label:
                    max_label = y_true[i]
                    max_label_idx = i
                i += 1
            end_idx = i - 1
            
            actual_events.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'peak_idx': max_label_idx,
                'start_time': timestamps[start_idx],
                'end_time': timestamps[end_idx],
                'peak_time': timestamps[max_label_idx]
            })
        else:
            i += 1
    
    # 收集所有時間差異
    time_diffs = []
    early_diffs = []  # 提前預測時間差
    late_diffs = []   # 延遲預測時間差
    
    # 匹配實際事件與預測事件
    for actual_event in actual_events:
        best_pred_event = None
        best_time_diff = float('inf')
        
        for pred_event in predicted_events:
            # 計算預測開始時間與實際事件發生時間的差距
            time_diff = pred_event['start_time'] - actual_event['peak_time']
            
            # 我們認為以下情況是有效的檢測
            if -WARNING_TIME <= time_diff <= 5:
                if abs(time_diff) < abs(best_time_diff):
                    best_time_diff = time_diff
                    best_pred_event = pred_event
        
        if best_pred_event is not None:
            time_diffs.append(best_time_diff)
            if best_time_diff <= 0:
                early_diffs.append(best_time_diff)  # 提前預測（負值）
            else:
                late_diffs.append(best_time_diff)   # 延遲預測（正值）
            
            # 標記已匹配
            best_pred_event['matched'] = True
    
    # 繪製時間差異分布圖
    plt.figure(figsize=(15, 10))
    
    # 1. 所有時間差異的直方圖和KDE
    plt.subplot(2, 2, 1)
    if time_diffs:
        plt.hist(time_diffs, bins=15, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
        plt.axvline(x=np.mean(time_diffs), color='green', linestyle='-', linewidth=2, 
                    label=f'平均值: {np.mean(time_diffs):.2f}秒')
    plt.title('所有檢測的時間差異分布')
    plt.xlabel('時間差異（秒）- 負值表示提前，正值表示延遲')
    plt.ylabel('頻率')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    # 2. 提前預測和延遲預測的分開直方圖
    plt.subplot(2, 2, 2)
    if early_diffs:
        plt.hist(early_diffs, bins=10, alpha=0.7, color='green', edgecolor='black', label='提前預測')
    if late_diffs:
        plt.hist(late_diffs, bins=5, alpha=0.7, color='orange', edgecolor='black', label='延遲預測')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
    if early_diffs:
        plt.axvline(x=np.mean(early_diffs), color='green', linestyle='-', linewidth=2, 
                    label=f'平均提前: {abs(np.mean(early_diffs)):.2f}秒')
    if late_diffs:
        plt.axvline(x=np.mean(late_diffs), color='orange', linestyle='-', linewidth=2, 
                    label=f'平均延遲: {np.mean(late_diffs):.2f}秒')
    plt.title('提前預測和延遲預測的時間差異分布')
    plt.xlabel('時間差異（秒）')
    plt.ylabel('頻率')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    # 3. 箱型圖比較提前預測和延遲預測
    plt.subplot(2, 2, 3)
    box_data = []
    labels = []
    
    if early_diffs:
        # 轉換為絕對值以便比較
        box_data.append([abs(diff) for diff in early_diffs])
        labels.append('提前預測')
    
    if late_diffs:
        box_data.append(late_diffs)
        labels.append('延遲預測')
    
    if box_data:
        plt.boxplot(box_data, tick_labels=labels, patch_artist=True, 
                    boxprops=dict(facecolor='lightblue'),
                    medianprops=dict(color='red'))
    plt.title('提前預測和延遲預測的時間差異箱型圖')
    plt.ylabel('時間差異（秒）- 絕對值')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 4. 目標時間差異比較
    plt.subplot(2, 2, 4)
    target_lead_time = 7  # 目標提前時間（秒）
    
    if early_diffs:
        # 計算與目標時間的偏差
        target_diffs = [abs(abs(diff) - target_lead_time) for diff in early_diffs]
        plt.hist(target_diffs, bins=10, alpha=0.7, color='purple', edgecolor='black')
        plt.axvline(x=np.mean(target_diffs), color='red', linestyle='-', linewidth=2, 
                    label=f'平均偏差: {np.mean(target_diffs):.2f}秒')
        
        # 3秒內算準確
        accurate_count = sum(1 for diff in target_diffs if diff <= 3)
        accuracy_rate = (accurate_count / len(target_diffs)) * 100 if target_diffs else 0
        
        plt.title(f'與目標提前時間的偏差分布\n({accuracy_rate:.1f}%在±3秒範圍內)')
        plt.xlabel('與目標時間的偏差（秒）')
        plt.ylabel('頻率')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
    else:
        plt.text(0.5, 0.5, '沒有提前預測數據', 
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=12)
        plt.title('與目標提前時間的偏差分布')
    
    # 添加統計信息
    early_count = len(early_diffs)
    late_count = len(late_diffs)
    total_count = len(time_diffs)
    
    avg_early_time = abs(np.mean(early_diffs)) if early_diffs else 0
    avg_late_time = np.mean(late_diffs) if late_diffs else 0
    avg_total_time = np.mean(time_diffs) if time_diffs else 0
    
    stats_text = f"統計資訊:\n"
    stats_text += f"總檢測數: {total_count}\n"
    stats_text += f"提前預測: {early_count} (平均提前 {avg_early_time:.2f}秒)\n"
    stats_text += f"延遲預測: {late_count} (平均延遲 {avg_late_time:.2f}秒)\n"
    
    if early_diffs:
        target_diffs = [abs(abs(diff) - target_lead_time) for diff in early_diffs]
        accurate_count = sum(1 for diff in target_diffs if diff <= 3)
        accuracy_rate = (accurate_count / len(target_diffs)) * 100 if target_diffs else 0
        stats_text += f"目標時間準確率: {accuracy_rate:.1f}% (±3秒內)"
    
    plt.figtext(0.5, 0.01, stats_text, fontsize=10, ha='center', 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # 保存圖表
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    time_diff_dist_path = os.path.join(LOG_DIR, 'prediction_time_diff_distribution.png')
    plt.savefig(time_diff_dist_path)
    print(f"預測時間差異分布圖已保存至: {time_diff_dist_path}")
    plt.close()

def generate_evaluation_summary(y_true, y_pred, timestamps, threshold=0.9):
    """生成簡潔的評估摘要報告"""
    # 直接使用與評估函數相同的二值化邏輯
    # 創建二值預測結果
    predictions = (y_pred >= threshold).astype(int)
    
    # 評估預測結果 - 不傳入閾值，因為該函數內部已固定使用0.9
    metrics, _, target_lead = evaluate_predictions(y_true, y_pred, timestamps, find_best_threshold=False)
    
    # 計算關鍵指標
    total_actual = metrics['detections'] + metrics['missed_events']
    detection_rate = metrics['detections'] / max(total_actual, 1) * 100
    early_rate = metrics['early_detections'] / max(metrics['detections'], 1) * 100
    late_rate = metrics['late_detections'] / max(metrics['detections'], 1) * 100
    
    avg_early_time = np.mean(metrics['early_time_diffs']) if metrics['early_time_diffs'] else 0
    avg_late_time = np.mean(metrics['late_time_diffs']) if metrics['late_time_diffs'] else 0
    
    # 計算時間準確率
    if metrics['early_time_diffs']:
        time_diff_seconds = abs(avg_early_time - target_lead)
        if time_diff_seconds <= 3:
            time_accuracy = 100.0
        else:
            time_accuracy = max(0, 100.0 * (1.0 - (time_diff_seconds - 3) / 7))
    else:
        time_accuracy = 0.0
    
    # 構建報告文本
    report = f"""
====== 離床預測模型評估摘要 ======

【基本統計】
- 閾值設定: {threshold:.2f} (注意: 評估函數固定使用0.9)
- 實際事件總數: {total_actual}
- 成功檢測數: {metrics['detections']} ({detection_rate:.2f}%)
- 漏報數: {metrics['missed_events']}
- 誤報數: {metrics['false_alarms']}

【時間預測】
- 提前預測: {metrics['early_detections']} ({early_rate:.2f}%)
- 延遲預測: {metrics['late_detections']} ({late_rate:.2f}%)
- 平均提前時間: {avg_early_time:.2f}秒 (目標: {target_lead:.1f}秒)
- 平均延遲時間: {avg_late_time:.2f}秒
- 時間準確率: {time_accuracy:.2f}% (±3秒內為100%)
"""
    
    # 保存報告
    report_path = os.path.join(LOG_DIR, 'evaluation_summary.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"評估摘要已保存至: {report_path}")
    return metrics

# 處理命令列參數
parser = argparse.ArgumentParser(description='CNN-LSTM模型用於預測離床事件')
parser.add_argument('--load-only', action='store_true', help='只載入模型預測，不重新訓練')
parser.add_argument('--threshold', type=float, default=0.8, help='預測閾值，默認為0.8')
parser.add_argument('--predict-new', action='store_true', help='只處理新資料並使用現有模型進行預測')
parser.add_argument('--prediction-dir', type=str, default=PREDICTION_DATA_DIR, help='指定預測資料夾路徑')
parser.add_argument('--prediction-file', type=str, default='', help='指定要預測的單一檔案路徑')
parser.add_argument('--loss-type', type=str, choices=['focal', 'time_diff', 'improved_time_diff', 'simplified_time_diff'], 
                   default='focal', help='選擇損失函數: focal (預設)、time_diff (時間差異)、improved_time_diff (改進時間差異)或simplified_time_diff (簡化時間差異)')
args = parser.parse_args()

# 在主程式中，修改資料載入部分
try:
    # 處理單一檔案預測模式
    if args.predict_new and args.prediction_file:
        # 確保模型存在
        final_model_path = os.path.join(LOG_DIR, FINAL_MODEL_PATH)
        if not os.path.exists(final_model_path):
            print(f"錯誤: 找不到現有模型 {final_model_path}，無法執行 --predict-new 模式")
            sys.exit(1)
        
        print(f"已啟用單一檔案預測模式: {args.prediction_file}")
        # 檢查檔案是否存在
        if not os.path.exists(args.prediction_file):
            print(f"錯誤: 找不到指定的檔案 {args.prediction_file}")
            sys.exit(1)
        
        # 處理檔案並生成cleaned_檔案
        print(f"開始處理檔案: {os.path.basename(args.prediction_file)}")
        
        # 載入和處理單一檔案 - 注意：此處加載的檔案仍會儲存在_data/training目錄
        original_data, original_labels, original_event_binary, original_feature_names = load_and_process_data(
            args.prediction_file,
            apply_balancing=False,  # 預測不需要平衡資料
            pos_to_neg_ratio=POS_TO_NEG_RATIO
        )
        
        # 修改：根據處理後的檔案路徑找到生成的cleaned_檔案
        training_cleaned_path = get_cleaned_data_path(args.prediction_file)
        print(f"已在訓練目錄生成處理後的檔案: {training_cleaned_path}")
        
        # 修改：在PREDICTION_DATA_DIR中創建相同的cleaned_檔案
        raw_filename = os.path.basename(args.prediction_file)
        prediction_cleaned_filename = f"cleaned_{raw_filename}"
        prediction_cleaned_path = os.path.join(PREDICTION_DATA_DIR, prediction_cleaned_filename)
        
        # 確保PREDICTION_DATA_DIR目錄存在
        os.makedirs(PREDICTION_DATA_DIR, exist_ok=True)
        
        # 複製檔案從_data/training到PREDICTION_DATA_DIR
        if os.path.exists(training_cleaned_path):
            # 讀取訓練目錄中的檔案
            training_df = pd.read_csv(training_cleaned_path)
            # 保存到預測目錄
            training_df.to_csv(prediction_cleaned_path, index=False)
            print(f"已複製處理後的檔案到預測目錄: {prediction_cleaned_path}")
        else:
            print(f"警告: 未找到訓練目錄中的處理後檔案: {training_cleaned_path}")
        
        # 載入模型
        print(f"載入現有模型: {final_model_path}")
        model = load_model(final_model_path, compile=False)
        print("模型載入完成")
        
        # 重塑資料以符合模型輸入格式
        X_pred = np.array(original_data).reshape((original_data.shape[0], original_data.shape[1], 1))
        
        # 進行預測
        print("開始預測...")
        pred_result = model.predict(X_pred)
        print(f"預測結果形狀: {np.shape(pred_result)}")
        
        # 壓平預測結果
        pred_result_flat = pred_result.flatten()
        
        # 計算最佳閾值
        adaptive_threshold = find_optimal_threshold(pred_result_flat)
        print(f"使用自適應閾值: {adaptive_threshold:.4f}")
        
        # 修改：同時將預測結果寫回兩個位置的清理過的檔案
        try:
            # 1. 處理預測目錄中的檔案
            if os.path.exists(prediction_cleaned_path):
                prediction_df = pd.read_csv(prediction_cleaned_path)
                
                # 添加預測結果欄位
                if len(pred_result_flat) >= len(prediction_df):
                    prediction_df['Predicted'] = (pred_result_flat[:len(prediction_df)] > adaptive_threshold).astype(int)
                    prediction_df['Predicted_Prob'] = pred_result_flat[:len(prediction_df)]
                else:
                    # 如果預測結果比原始資料短，只修改有預測結果的部分
                    print(f"警告: 預測結果長度({len(pred_result_flat)})小於原始資料長度({len(prediction_df)})")
                    prediction_df.loc[:len(pred_result_flat)-1, 'Predicted'] = (pred_result_flat > adaptive_threshold).astype(int)
                    prediction_df.loc[:len(pred_result_flat)-1, 'Predicted_Prob'] = pred_result_flat
                
                # 保存修改後的檔案到預測目錄
                prediction_df.to_csv(prediction_cleaned_path, index=False)
                print(f"預測結果已寫入預測目錄檔案: {prediction_cleaned_path}")
            else:
                print(f"警告: 預測目錄中的檔案不存在: {prediction_cleaned_path}")
            
            # 2. 也處理訓練目錄中的檔案，保持向後兼容
            if os.path.exists(training_cleaned_path):
                training_df = pd.read_csv(training_cleaned_path)
                
                # 添加預測結果欄位
                if len(pred_result_flat) >= len(training_df):
                    training_df['Predicted'] = (pred_result_flat[:len(training_df)] > adaptive_threshold).astype(int)
                    training_df['Predicted_Prob'] = pred_result_flat[:len(training_df)]
                else:
                    print(f"警告: 預測結果長度({len(pred_result_flat)})小於原始資料長度({len(training_df)})")
                    training_df.loc[:len(pred_result_flat)-1, 'Predicted'] = (pred_result_flat > adaptive_threshold).astype(int)
                    training_df.loc[:len(pred_result_flat)-1, 'Predicted_Prob'] = pred_result_flat
                
                # 保存修改後的檔案到訓練目錄
                training_df.to_csv(training_cleaned_path, index=False)
                print(f"預測結果也已寫入訓練目錄檔案: {training_cleaned_path}")
            
            # 另存一份結果至LOG_DIR
            base_filename = os.path.splitext(os.path.basename(prediction_cleaned_path))[0]
            result_path = os.path.join(LOG_DIR, f"{base_filename}_prediction.csv")
            
            # 使用預測目錄的DataFrame儲存至LOG_DIR
            if os.path.exists(prediction_cleaned_path):
                prediction_df.to_csv(result_path, index=False)
            else:
                training_df.to_csv(result_path, index=False)
                
            print(f"預測結果副本已保存至: {result_path}")
            
            # 繪製預測結果圖 - 使用預測目錄的資料
            plt.figure(figsize=(15, 6))
            
            # 選擇要繪圖的DataFrame
            plot_df = prediction_df if os.path.exists(prediction_cleaned_path) else training_df
            
            plt.plot(range(len(plot_df)), plot_df['event_binary'], 'b-', alpha=0.5, label='實際值')
            plt.plot(range(len(plot_df)), plot_df['Predicted_Prob'], 'r-', alpha=0.5, label='預測值')
            plt.axhline(y=adaptive_threshold, color='g', linestyle='--', label=f'閾值 ({adaptive_threshold:.4f})')
            plt.title(f'預測結果 - {os.path.basename(prediction_cleaned_path)}')
            plt.xlabel('資料索引')
            plt.ylabel('值')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plot_file = os.path.join(LOG_DIR, f"{base_filename}_prediction.png")
            plt.savefig(plot_file)
            plt.close()
            print(f"預測結果圖表已保存至: {plot_file}")
            
            # 顯示預測摘要
            positives = np.sum(pred_result_flat > adaptive_threshold)
            print(f"\n預測結果摘要:")
            print(f"總筆數: {len(pred_result_flat)}")
            print(f"預測值 > {adaptive_threshold:.4f} 的筆數: {positives}")
            print(f"預測值的範圍: {np.min(pred_result_flat)} 至 {np.max(pred_result_flat)}")
            
            print("\n處理完成!")
        except Exception as e:
            print(f"處理預測結果時發生錯誤: {e}")
        
        # 完成單一檔案預測模式，退出程式
        sys.exit(0)
        
    # 獲取所有符合條件的檔案
    INPUT_DATA_PATHS = glob.glob(os.path.join(INPUT_DATA_DIR, INPUT_DATA_PATTERN))
    if not INPUT_DATA_PATHS:
        print(f"錯誤: 在 {INPUT_DATA_DIR} 目錄下未找到任何符合 {INPUT_DATA_PATTERN} 的檔案")
        sys.exit(1)
    
    # 獲取預測用的資料檔案
    PREDICTION_DATA_DIR = args.prediction_dir  # 使用命令列參數傳入的路徑
    os.makedirs(PREDICTION_DATA_DIR, exist_ok=True)  # 確保預測資料夾存在
    PREDICTION_DATA_PATHS = glob.glob(os.path.join(PREDICTION_DATA_DIR, PREDICTION_DATA_PATTERN))
    print(f"找到 {len(PREDICTION_DATA_PATHS)} 個預測用資料檔案:")
    for i, path in enumerate(PREDICTION_DATA_PATHS):
        print(f"  {i+1}. {os.path.basename(path)}")
    
    print(f"找到 {len(INPUT_DATA_PATHS)} 個訓練資料檔案:")
    for i, path in enumerate(INPUT_DATA_PATHS):
        print(f"  {i+1}. {os.path.basename(path)}")
    
    # 初始化存儲所有資料的陣列
    all_sequences = []
    all_labels = []
    all_event_binary = []
    feature_names = None
    
    # 逐一處理每個檔案
    for i, file_path in enumerate(INPUT_DATA_PATHS):
        print(f"\n處理檔案 {i+1}/{len(INPUT_DATA_PATHS)}: {os.path.basename(file_path)}")
        
        # 使用總和值進行訓練 
        sequences, labels, event_binary, current_feature_names = load_and_process_data(
            file_path,
            apply_balancing=APPLY_BALANCING,
            pos_to_neg_ratio=POS_TO_NEG_RATIO
        )
        
        # 第一個檔案時設定feature_names
        if feature_names is None:
            feature_names = current_feature_names
        # 驗證各檔案的feature_names一致性
        elif not np.array_equal(feature_names, current_feature_names):
            print(f"警告: 檔案 {os.path.basename(file_path)} 的特徵名稱與之前的不一致")
            print(f"預期: {feature_names}")
            print(f"實際: {current_feature_names}")
            # 繼續處理，但可能會導致問題
        
        # 將當前檔案的資料添加到總資料集中
        all_sequences.append(sequences)
        all_labels.append(labels)
        all_event_binary.append(event_binary)
        
        print(f"檔案 {os.path.basename(file_path)} 中包含 {np.sum(event_binary)} 個離床事件")
    
    # 合併所有檔案的資料
    X = np.vstack(all_sequences)
    y = np.concatenate(all_labels)
    event_binary_all = np.concatenate(all_event_binary)
    
    print(f"\n合併後資料統計:")
    print(f"總序列形狀: {X.shape}")
    print(f"總標籤形狀: {y.shape}")
    print(f"總共有 {np.sum(event_binary_all)} 個離床事件")
    print(f"特徵名稱: {feature_names}")
    print(f"注意: 已移除標籤特徵(OnBed_Status和event_binary)作為輸入，僅使用以下八個感測器相關特徵作為模型輸入:")
    for i, feature in enumerate(feature_names):
        print(f"  {i+1}. {feature}")

    # 不再分割訓練和測試集，直接使用所有資料
    X_all = X.reshape((X.shape[0], X.shape[1], 1))
    y_all = y

    # 創建日誌目錄
    os.makedirs(LOG_DIR, exist_ok=True)

    # 定義檔案路徑
    final_model_path = os.path.join(LOG_DIR, FINAL_MODEL_PATH)

    # 生成完整的時間戳序列 - 移到這裡，確保在設定callbacks前定義
    timestamps = np.arange(len(X))  # 使用完整數據集的長度
    
    if args.predict_new:
        # 只處理新資料並使用現有模型進行預測
        if os.path.exists(final_model_path):
            print(f"載入現有模型進行新資料預測: {final_model_path}")
            model = load_model(final_model_path, compile=False)
            print("模型載入完成，跳過訓練過程")
            
            # 如果PREDICTION_DATA_PATHS為空，提示用戶
            if not PREDICTION_DATA_PATHS:
                print(f"警告: 在預測資料夾中未找到任何符合 {PREDICTION_DATA_PATTERN} 的檔案")
                print(f"請使用 --prediction-file 參數指定要預測的單一檔案，或在 {PREDICTION_DATA_DIR} 目錄中放入原始資料檔案")
                sys.exit(1)
                
            # 進入預測模式，不需要訓練資料
            # 所有訓練資料相關的處理會被跳過
            all_sequences = []
            all_labels = []
            all_event_binary = []
            X_all = np.array([[0]])  # 創建一個空模型輸入，因為不需要訓練
            y_all = np.array([0])    # 創建一個空標籤陣列
        else:
            print(f"錯誤: 找不到現有模型 {final_model_path}，無法執行 --predict-new 模式")
            sys.exit(1)
    elif args.load_only and os.path.exists(final_model_path):
        # 只載入模型預測
        print(f"載入現有模型: {final_model_path}")
        model = load_model(final_model_path, compile=False)
        print("模型載入完成，跳過訓練過程")
    else:
        # 訓練新模型
        print("建立並訓練新模型...")
        print(f"使用損失函數類型: {args.loss_type}")
        model = build_model((X_all.shape[1], X_all.shape[2]), loss_type=args.loss_type)
        
        # 設定回調函數
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=final_model_path,
                save_best_only=True,
                monitor='loss',
                mode='min'
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=5,
                restore_best_weights=True
            ),
            EventEvaluationCallback(validation_data=(X_all, y_all), timestamps=timestamps)
        ]
        
        # 訓練模型
        history = model.fit(
            X_all, y_all,
            epochs=2,
            batch_size=32,
            callbacks=callbacks,
            class_weight={0: 1, 1: 400},  # 使用更高的權重比例，專注於離床事件
            verbose=1
        )
        
        print("模型訓練完成")
        
        # 儲存最終模型
        model.save(final_model_path)
        print(f"模型已儲存至: {final_model_path}")

    # 預測所有資料 - 修改為僅使用最後一個檔案進行預測展示
    # 如果是 --predict-new 模式且沒有訓練資料，跳過這部分處理
    if args.predict_new and not all_sequences:
        print("\n進入單純預測模式，跳過訓練資料評估部分...")
    else:
        print("\n使用最後一個檔案進行預測示範...")
        last_file_sequences = all_sequences[-1]
        last_file_labels = all_labels[-1]
        
        # 將最後一個檔案的資料重新整理為模型輸入格式
        X_last = np.array(last_file_sequences).reshape((last_file_sequences.shape[0], last_file_sequences.shape[1], 1))
        y_last = np.array(last_file_labels)
        
        # 使用訓練後的模型對最後一個檔案進行預測
        all_pred = model.predict(X_last)
        print(f"預測結果形狀: {np.shape(all_pred)}")

        # 將預測結果壓平
        all_pred_flat = all_pred.flatten()
        
        # 添加調試信息來檢查數組長度不一致問題
        print(f"all_pred_flat 長度: {len(all_pred_flat)}")
        print(f"y_last 長度: {len(y_last)}")
        print(f"range(len(all_pred_flat)) 長度: {len(range(len(all_pred_flat)))}")
        
        # 將原始標籤和預測結果保存到CSV - 修正數組長度不一致問題
        # 創建只包含索引和預測值的DataFrame
        results_df = pd.DataFrame({
            'Index': range(len(all_pred_flat)),
            'Predicted': all_pred_flat
        })
        
        # 如果y_last長度與all_pred_flat不同，使用NaN填充或只使用可用部分
        if len(y_last) < len(all_pred_flat):
            print(f"警告: 實際標籤數量({len(y_last)})少於預測結果數量({len(all_pred_flat)})")
            # 創建與all_pred_flat相同長度的數組，前len(y_last)個值使用y_last，其餘為NaN
            actual_values = np.full(len(all_pred_flat), np.nan)
            actual_values[:len(y_last)] = y_last
            results_df['Actual'] = actual_values
        else:
            # 如果y_last更長或長度相同，只使用前len(all_pred_flat)個值
            results_df['Actual'] = y_last[:len(all_pred_flat)]
        
        output_file = os.path.join(LOG_DIR, "all_predictions.csv")
        results_df.to_csv(output_file, index=False)
        print(f"預測結果已保存至: {output_file}")

        cleaned_data_path = get_cleaned_data_path(INPUT_DATA_PATHS[-1])
        PROCESSED_DATA_PATH = cleaned_data_path.replace('.csv', '_processed.csv')

        print(f"PROCESSED_DATA_PATH: {PROCESSED_DATA_PATH}")
        if PROCESSED_DATA_PATH and os.path.exists(os.path.dirname(PROCESSED_DATA_PATH)):
            # 檢查是否需要匹配已存在的CSV檔案的格式
            try:
                # 讀取原始處理好的檔案
                original_df = pd.read_csv(PROCESSED_DATA_PATH)
                # 確保我們有足夠的預測結果
                if len(all_pred_flat) >= len(original_df):
                    # 使用自適應閾值檢測來確定最佳閾值
                    adaptive_threshold = find_optimal_threshold(all_pred_flat[:len(original_df)])
                    print(f"使用自適應閾值: {adaptive_threshold:.4f} (原始閾值: {args.threshold})")
                    
                    # 添加預測結果欄位（二值化結果）
                    original_df['Predicted'] = (all_pred_flat[:len(original_df)] > adaptive_threshold).astype(int)
                    # 添加預測機率值欄位
                    original_df['Predicted_Prob'] = all_pred_flat[:len(original_df)]
                    # 保存回原始檔案
                    original_df.to_csv(PROCESSED_DATA_PATH, index=False)
                    print(f"預測結果已添加到原始處理檔案: {PROCESSED_DATA_PATH}")
                else:
                    print(f"警告: 預測結果數量({len(all_pred_flat)})少於原始檔案行數({len(original_df)})")
                    # 仍然保存results_df到PROCESSED_DATA_PATH
                    results_df.to_csv(PROCESSED_DATA_PATH, index=False)
            except Exception as e:
                print(f"處理原始檔案時發生錯誤: {e}, 直接保存預測結果")
                results_df.to_csv(PROCESSED_DATA_PATH, index=False)
        else:
            print(f"警告: 處理後的數據路徑不存在或無效: {PROCESSED_DATA_PATH}")
            # 建立目錄（如果需要）
            os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
            results_df.to_csv(PROCESSED_DATA_PATH, index=False)
        
        print(f"預測結果已保存至: {PROCESSED_DATA_PATH}")

        # 輸出預測結果摘要
        print(f"\n預測結果摘要:")
        print(f"總筆數: {len(all_pred_flat)}")
        adaptive_threshold = find_optimal_threshold(all_pred_flat)
        print(f"預測值 > {adaptive_threshold:.4f} 的筆數: {np.sum(all_pred_flat > adaptive_threshold)}")
        print(f"預測值的範圍: {np.min(all_pred_flat)} 至 {np.max(all_pred_flat)}")
        print(f"預測值中最高的前五筆: {np.sort(all_pred_flat)[-5:]}")

        # 繪製預測結果
        plt.figure(figsize=(15, 6))
        plt.plot(results_df['Index'], results_df['Actual'], 'b-', alpha=0.5, label='實際值')
        plt.plot(results_df['Index'], results_df['Predicted'], 'r-', alpha=0.5, label='預測值')
        plt.axhline(y=adaptive_threshold, color='g', linestyle='--', label=f'閾值 ({adaptive_threshold:.4f})')
        plt.title('所有資料的預測結果')
        plt.xlabel('資料索引')
        plt.ylabel('值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_file = os.path.join(LOG_DIR, "all_predictions_plot.png")
        plt.savefig(plot_file)
        plt.close()
        print(f"預測結果圖表已保存至: {plot_file}")

    # 預測資料 - 修改為預測指定資料夾中的所有檔案
    print("\n開始預測資料...")
    
    # 檢查是否有預測資料
    if len(PREDICTION_DATA_PATHS) == 0:
        print("警告: 預測資料夾中沒有資料，將使用最後一個訓練檔案進行示範預測")
        # 使用最後一個訓練檔案作為預測示範
        prediction_paths = [INPUT_DATA_PATHS[-1]]
    else:
        prediction_paths = PREDICTION_DATA_PATHS
    
    # 對每個預測檔案進行處理
    for pred_index, pred_file_path in enumerate(prediction_paths):
        print(f"\n預測檔案 {pred_index+1}/{len(prediction_paths)}: {os.path.basename(pred_file_path)}")
        
        # 載入預測用資料
        try:
            # 檢查檔案名稱是否已包含 "cleaned_" 前綴
            basename = os.path.basename(pred_file_path)
            is_already_cleaned = basename.startswith("cleaned_")
            
            # 如果是已清理檔案，直接處理；否則按原邏輯處理
            if is_already_cleaned:
                print(f"檢測到已清理的檔案: {basename}")
                dataset = pd.read_csv(pred_file_path)
                print(f"數據集形狀: {dataset.shape}")
                print(f"數據集列: {dataset.columns.tolist()}")
                
                # 確保所需的欄位都存在
                required_columns = [f'Channel_{i}_Raw' for i in range(1, 7)]
                required_columns.extend(['Raw_sum', 'Noise_max', 'OnBed_Status'])
                
                # 檢查是否有event_binary欄位，沒有則添加
                if 'event_binary' not in dataset.columns:
                    print("未找到event_binary欄位，將生成該欄位")
                    dataset = detect_bed_events(dataset)
                
                required_columns.append('event_binary')
                
                if not all(col in dataset.columns for col in required_columns):
                    missing_cols = [col for col in required_columns if col not in dataset.columns]
                    raise ValueError(f"缺少必要的列: {missing_cols}")
                
                # 選取所需欄位
                df_features = dataset[required_columns].copy()
                
                # 整理為模型輸入格式
                pred_sequences = df_features.values.astype('float64')
                pred_labels = dataset['event_binary'].values
                pred_feature_names = df_features.columns.tolist()
                
                # 確保序列長度符合要求
                if len(pred_sequences) < 86401:
                    # 使用零填充
                    padding = np.zeros((86401 - len(pred_sequences), pred_sequences.shape[1]))
                    pred_sequences = np.vstack([pred_sequences, padding])
                    pred_labels = np.pad(pred_labels, (0, 86401 - len(pred_labels)), 'constant')
                    print(f"已填充序列長度: {len(pred_sequences)}")
                elif len(pred_sequences) > 86401:
                    # 直接截斷超過的部分
                    print(f"序列長度超過86400，直接截斷多餘部分")
                    pred_sequences = pred_sequences[:86401]
                    pred_labels = pred_labels[:86401]
                    print(f"已截斷序列長度: {len(pred_sequences)}")
            else:
                # 使用原有邏輯載入和處理
                pred_sequences, pred_labels, pred_event_binary, pred_feature_names = load_and_process_data(
                    pred_file_path,
                    apply_balancing=False,  # 預測不需要平衡資料
                    pos_to_neg_ratio=POS_TO_NEG_RATIO
                )
            
            # 重新整理為模型輸入格式
            X_pred = np.array(pred_sequences).reshape((pred_sequences.shape[0], pred_sequences.shape[1], 1))
            y_pred_actual = np.array(pred_labels)
            
            # 使用訓練後的模型進行預測
            pred_result = model.predict(X_pred)
            print(f"預測結果形狀: {np.shape(pred_result)}")
            
            # 將預測結果壓平
            pred_result_flat = pred_result.flatten()
            
            # 檢查數組長度
            print(f"預測結果長度: {len(pred_result_flat)}")
            print(f"實際標籤長度: {len(y_pred_actual)}")
            
            # 創建結果DataFrame
            pred_results_df = pd.DataFrame({
                'Index': range(len(pred_result_flat)),
                'Predicted': pred_result_flat
            })
            
            # 處理可能的長度不一致問題
            if len(y_pred_actual) < len(pred_result_flat):
                print(f"警告: 實際標籤數量({len(y_pred_actual)})少於預測結果數量({len(pred_result_flat)})")
                actual_values = np.full(len(pred_result_flat), np.nan)
                actual_values[:len(y_pred_actual)] = y_pred_actual
                pred_results_df['Actual'] = actual_values
            else:
                pred_results_df['Actual'] = y_pred_actual[:len(pred_result_flat)]
            
            # 計算預測摘要
            adaptive_threshold = find_optimal_threshold(pred_result_flat)
            print(f"使用自適應閾值: {adaptive_threshold:.4f} (原始閾值: {args.threshold})")
            
            positives = np.sum(pred_result_flat > adaptive_threshold)
            print(f"\n預測結果摘要 ({os.path.basename(pred_file_path)}):")
            print(f"總筆數: {len(pred_result_flat)}")
            print(f"預測值 > {adaptive_threshold:.4f} 的筆數: {positives}")
            print(f"預測值的範圍: {np.min(pred_result_flat)} 至 {np.max(pred_result_flat)}")
            print(f"預測值中最高的前五筆: {np.sort(pred_result_flat)[-5:]}")
            
            # 建立檔案名稱（基於原始檔名）
            base_filename = os.path.splitext(os.path.basename(pred_file_path))[0]
            
            # 保存預測結果CSV
            pred_output_file = os.path.join(LOG_DIR, f"predictions_{base_filename}.csv")
            pred_results_df.to_csv(pred_output_file, index=False)
            print(f"預測結果已保存至: {pred_output_file}")
            
            # 直接將預測結果寫回原始檔案
            try:
                # 檢查原始資料集長度與預測結果長度
                if len(dataset) <= len(pred_result_flat):
                    # 添加預測結果欄位（二值化和機率值）
                    dataset['Predicted'] = (pred_result_flat[:len(dataset)] > adaptive_threshold).astype(int)
                    dataset['Predicted_Prob'] = pred_result_flat[:len(dataset)]
                    # 保存修改後的資料集
                    dataset.to_csv(pred_file_path, index=False)
                    print(f"預測結果已寫回原始檔案: {pred_file_path}")
                else:
                    print(f"警告: 原始檔案行數({len(dataset)})大於預測結果數量({len(pred_result_flat)})")
                    print("將只更新前 {len(pred_result_flat)} 筆資料")
                    dataset.loc[:len(pred_result_flat)-1, 'Predicted'] = (pred_result_flat > adaptive_threshold).astype(int)
                    dataset.loc[:len(pred_result_flat)-1, 'Predicted_Prob'] = pred_result_flat
                    dataset.to_csv(pred_file_path, index=False)
                    print(f"預測結果已部分寫回原始檔案: {pred_file_path}")
            except Exception as e:
                print(f"寫回原始檔案時發生錯誤: {e}")
            # 繪製預測結果圖
            plt.figure(figsize=(15, 6))
            plt.plot(pred_results_df['Index'], pred_results_df['Actual'], 'b-', alpha=0.5, label='實際值')
            plt.plot(pred_results_df['Index'], pred_results_df['Predicted'], 'r-', alpha=0.5, label='預測值')
            plt.axhline(y=adaptive_threshold, color='g', linestyle='--', label=f'閾值 ({adaptive_threshold:.4f})')
            plt.title(f'預測結果 - {os.path.basename(pred_file_path)}')
            plt.xlabel('資料索引')
            plt.ylabel('值')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plot_file = os.path.join(LOG_DIR, f"predictions_plot_{base_filename}.png")
            plt.savefig(plot_file)
            plt.close()
            print(f"預測結果圖表已保存至: {plot_file}")
            
        except Exception as e:
            print(f"預測檔案 {os.path.basename(pred_file_path)} 時發生錯誤: {e}")
            continue

except Exception as e:
    print(f"數據處理錯誤: {e}")
    raise



