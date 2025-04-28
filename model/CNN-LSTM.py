import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, LSTM, BatchNormalization, Multiply, Bidirectional, Add, Concatenate
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

# 設定參數
WINDOW_SIZE = 15  # 15秒的窗口
OVERLAP = 0.8    # 80% 重疊
STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP))  # 滑動步長

# 修改預警時間設定
WARNING_TIME = 15  # 設定單一預警時間（秒）

# INPUT_DATA_PATH = "./_data/pyqt_viewer/SPS2025PA000146_20250406_04_20250407_04_data.csv"
INPUT_DATA_PATH = "./_data/pyqt_viewer/SPS2024PA000329_20250420_03_20250421_04_data.csv"
TRAINING_LOG_PATH = "training_test_sum.csv"
FINAL_MODEL_PATH = "final_model_test_sum.keras"
TRAINING_HISTORY_PATH = "training_history_test_sum.png"
LOG_DIR = "./_logs/bed_monitor_test_sum"

# 確保LOG_DIR和其他必要目錄存在
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs("./_data/training", exist_ok=True)

FIND_BEST_THRESHOLD = False
SILENCE_TIME = 180 
APPLY_BALANCING = False
POS_TO_NEG_RATIO = float('inf') # 設為無限大表示不進行下採樣


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
        self.threshold = 0.9  # 使用0.9作為閾值
        self.target_lead_time = 7  # 目標提前時間（秒）
        
    def on_epoch_end(self, epoch, logs=None):
        x_val, y_val = self.validation_data
        y_pred = self.model.predict(x_val)
        metrics, threshold, target_lead = evaluate_predictions(y_val, y_pred, self.timestamps, find_best_threshold=False)
        
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
    # 將sequences轉換回DataFrame，使用原始欄位名稱
    df = pd.DataFrame(sequences, columns=feature_names)
    
    # 保存CSV格式，使用原始欄位名稱
    df.to_csv(cleaned_data_path.replace('.csv', '_processed.csv'), index=False)

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
                feature_names=feature_names)  # 使用保存的欄位名稱
        
    print(f"序列資料已保存至: {sequences_path}")

def detect_bed_events(df):
    """
    檢測離床事件，並在事件發生後的靜默時間內不檢測新事件。
    在原始 DataFrame 中添加 event_binary 欄位，值為 0 或 1
    
    返回:
    - 新增了 event_binary 欄位的 DataFrame
    """
    events = []
    event_binary = np.zeros(len(df))  # 創建一個全0陣列，長度與原始資料相同
    status_changes = df['OnBed_Status'].diff()
    last_event_time = -SILENCE_TIME  # 初始化為負值，確保第一個事件可以被檢測
    
    # 找出所有離床事件
    for idx in range(1, len(df)):
        # 檢查是否在靜默時間內
        if idx - last_event_time < SILENCE_TIME:
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

def balance_samples(df_features, event_labels, window_size=WINDOW_SIZE, step_size=STEP_SIZE, pos_to_neg_ratio=2):
    """
    收集和平衡正負樣本的比例
    
    參數:
    - df_features: 特徵資料框
    - event_labels: 事件標籤列表
    - window_size: 滑動窗口大小
    - step_size: 滑動步長
    - pos_to_neg_ratio: 負樣本數量與正樣本數量的比例，默認為2
    
    返回:
    - 平衡後的特徵序列列表和標籤列表
    """
    # 分別收集正負樣本（正樣本包括所有標籤 >= 0.5 的樣本）
    positive_sequences = []
    positive_labels = []
    negative_sequences = []
    negative_labels = []
    
    for i in range(0, len(df_features) - window_size + 1, step_size):
        window = df_features.iloc[i:(i + window_size)]
        label = event_labels[i + window_size - 1]
        
        if label >= 0.5:  # 所有標籤 >= 0.5 都視為正樣本
            positive_sequences.append(window.values.astype('float64'))
            positive_labels.append(label)
        else:
            negative_sequences.append(window.values.astype('float64'))
            negative_labels.append(label)
    
    print(f"收集到正樣本: {len(positive_sequences)}個, 負樣本: {len(negative_sequences)}個")
    
    # 對負樣本進行下採樣
    neg_sample_size = len(positive_sequences) * pos_to_neg_ratio
    
    # 只有當負樣本數量超過所需數量時才進行下採樣
    if len(negative_sequences) > neg_sample_size:
        indices = np.random.choice(len(negative_sequences), neg_sample_size, replace=False)
        negative_sequences = [negative_sequences[i] for i in indices]
        negative_labels = [negative_labels[i] for i in indices]
        print(f"下採樣後的負樣本數量: {len(negative_sequences)}個")
    
    # 合併正負樣本
    sequences = positive_sequences + negative_sequences
    labels = positive_labels + negative_labels
    
    # 打亂順序
    combined = list(zip(sequences, labels))
    np.random.shuffle(combined)
    sequences, labels = zip(*combined)
    
    print(f"最終樣本數量: {len(sequences)}個")
    return sequences, labels

def create_sequences(df, cleaned_data_path, apply_balancing=APPLY_BALANCING, pos_to_neg_ratio=POS_TO_NEG_RATIO):
    """修改後的序列創建函數，使用完整時間序列"""
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

    required_columns = [f'Channel_{i}_Raw' for i in range(1, 7)]
    required_columns.append('Raw_sum')
    required_columns.append('Noise_max')
    required_columns.append('OnBed_Status')
    required_columns.append('event_binary')  # 添加新的 event_binary 欄位
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"缺少必要的列: {[col for col in required_columns if col not in df.columns]}")
    df_features = df[required_columns].copy()
    
    # 存一份CSV
    df_features.to_csv(cleaned_data_path, index=False)
    print(f"已保存CSV: {cleaned_data_path}")
    
    # 保存原始欄位名稱
    feature_names = df_features.columns.tolist()
    
    # 轉換為數組時保留欄位名稱資訊
    sequences = df_features.values.astype('float64')
    labels = df['event_binary'].values
    
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
        thresholds = np.arange(0.3, 0.9, 0.05)  # 修改閾值範圍，包含更高值
        best_threshold = 0.9  # 默認最佳閾值設為0.9
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
        # 直接使用0.9作為閾值，而非之前的0.5
        metrics, score, target_lead_time = evaluate_with_threshold(0.9)
        return metrics, 0.9, target_lead_time

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

# 更先進的時間差異損失函數 - 使用純時間差計算
def pure_time_difference_loss(y_true, y_pred, target_lead_time=WARNING_TIME/2):
    """
    適應漸進式標籤的時間差異損失函數
    
    參數:
    - y_true: 實際標籤（現在是漸進式的，從0.5到1.0）
    - y_pred: 預測概率
    - target_lead_time: 目標提前時間（秒）
    """
    # 轉換為張量
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # 基礎損失 - 改為均方誤差，更適合漸進式標籤
    # 修正：使用正確的 TensorFlow API
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # 區分不同預測情況
    
    # 1. 關鍵離床區域（標籤>=0.7) - 這些是最接近離床事件的時間點
    critical_event_mask = tf.cast(tf.greater_equal(y_true, 0.7), tf.float32)
    
    # 2. 預警區域 (0.5 <= 標籤 < 0.7) - 這些是離床前的預警時間
    warning_mask = tf.cast(
        tf.logical_and(
            tf.greater_equal(y_true, 0.5),
            tf.less(y_true, 0.7)
        ), 
        tf.float32
    )
    
    # 3. 非事件區域 (標籤<0.5) - 正常區域，沒有離床事件
    normal_mask = tf.cast(tf.less(y_true, 0.5), tf.float32)
    
    # 調整閾值為0.9
    prediction_threshold = 0.9
    
    # 漏報懲罰 - 針對關鍵離床區域中預測值低於閾值的情況
    # 這是最嚴重的錯誤，因為我們錯過了關鍵的離床事件
    critical_miss_penalty = tf.where(
        tf.logical_and(
            tf.less(y_pred, prediction_threshold),   # 預測值低於閾值
            tf.greater_equal(y_true, 0.7)            # 真實值高（關鍵區域）
        ),
        tf.ones_like(y_true) * 20.0,                # 極高懲罰
        tf.ones_like(y_true)
    )
    
    # 預警區漏報 - 較輕的懲罰
    warning_miss_penalty = tf.where(
        tf.logical_and(
            tf.less(y_pred, prediction_threshold),                   # 預測值低於閾值
            tf.logical_and(                                         # 預警區域
                tf.greater_equal(y_true, 0.5),
                tf.less(y_true, 0.7)
            )
        ),
        tf.ones_like(y_true) * 5.0,                                # 中等懲罰
        tf.ones_like(y_true)
    )
    
    # 誤報懲罰 - 標籤<0.5但預測>閾值的情況
    # 這也很嚴重，但不如漏報
    false_alarm_penalty = tf.where(
        tf.logical_and(
            tf.greater_equal(y_pred, prediction_threshold),  # 高預測值
            tf.less(y_true, 0.5)                           # 非事件區域
        ),
        tf.ones_like(y_true) * 8.0,                        # 高懲罰
        tf.ones_like(y_true)
    )
    
    # 輕度誤報 - 標籤<0.5但0.5<預測<閾值的情況
    # 這是較輕的誤報
    minor_false_alarm_penalty = tf.where(
        tf.logical_and(
            tf.logical_and(
                tf.greater_equal(y_pred, 0.5),
                tf.less(y_pred, prediction_threshold)
            ),
            tf.less(y_true, 0.5)
        ),
        tf.ones_like(y_true) * 2.0,  # 較輕懲罰
        tf.ones_like(y_true)
    )
    
    # 成功預測獎勵 - 對於關鍵區域的高預測值
    # 我們希望在關鍵時刻得到高預測值
    success_reward = tf.where(
        tf.logical_and(
            tf.greater_equal(y_pred, prediction_threshold),
            tf.greater_equal(y_true, 0.7)
        ),
        tf.ones_like(y_true) * 0.2,  # 減輕損失（獎勵）
        tf.ones_like(y_true)
    )
    
    # 計算總體加權損失
    weighted_loss = mse * critical_miss_penalty * warning_miss_penalty * false_alarm_penalty * minor_false_alarm_penalty * success_reward
    
    return tf.reduce_mean(weighted_loss)

def build_model(input_shape):
    input_layer = Input(shape=input_shape)
    
    # 修改 CNN 層的配置
    conv1 = Conv1D(64, kernel_size=1, padding='same', activation='relu')(input_layer)
    conv1 = BatchNormalization()(conv1)
    conv2 = Conv1D(64, kernel_size=1, padding='same', activation='relu')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Add()([conv1, conv2])
    
    # 移除 MaxPooling1D 層，因為我們只有一個時間步長
    x = Dropout(0.25)(conv2)
    
    # LSTM層保持不變
    x = Bidirectional(LSTM(320, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    
    x = Bidirectional(LSTM(160))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    
    # 確保密集層的維度匹配
    dense1 = Dense(64, activation='relu')(x)
    dense1 = BatchNormalization()(dense1)
    dense2 = Dense(64, activation='relu')(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Add()([dense1, dense2])
    
    output_layer = Dense(1, activation='sigmoid')(dense2)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    # 使用純時間差異損失函數
    def loss_wrapper(y_true, y_pred):
        return pure_time_difference_loss(y_true, y_pred, target_lead_time=WARNING_TIME/2)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00025),
        loss=loss_wrapper,
        metrics=['accuracy', 'recall', 'precision']
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

# 修改原本的數據載入部分
try:
    # 使用總和值進行訓練 
    sequences, labels, event_binary, feature_names = load_and_process_data(
        INPUT_DATA_PATH,
        apply_balancing=APPLY_BALANCING,
        pos_to_neg_ratio=POS_TO_NEG_RATIO
    )

    print(f"sequences shape: {np.shape(sequences)}")
    print(f"labels shape: {np.shape(labels)}")
    print(f"event_binary shape: {np.shape(event_binary)}")
    print(f"feature_names: {feature_names}")   
    print(f"總共有 {np.sum(event_binary)} 個離床事件") 

    sys.exit()

    X = np.array(sequences)
    y = np.array(labels)
    print(f"特徵形狀: {X.shape}")
    print(f"標籤形狀: {y.shape}")
    print(f"特徵名稱: {feature_names}")

    # 分割訓練和測試集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=80)

    # 重塑數據為 (samples, timesteps, features) 格式
    # 使用 WINDOW_SIZE 作為時間步長
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # 創建日誌目錄
    os.makedirs(LOG_DIR, exist_ok=True)

    # 定義檔案路徑
    model_checkpoint_path = os.path.join(LOG_DIR, 'model-{epoch:02d}-{val_accuracy:.4f}.keras')
    training_log_path = os.path.join(LOG_DIR, TRAINING_LOG_PATH)
    final_model_path = os.path.join(LOG_DIR, FINAL_MODEL_PATH)
    training_history_path = os.path.join(LOG_DIR, TRAINING_HISTORY_PATH)

    # 建立模型
    model = build_model((X_train.shape[1], X_train.shape[2]))

    # 生成完整的時間戳序列
    timestamps = np.arange(len(X))  # 使用完整數據集的長度

    # 分割訓練集和測試集的時間戳
    train_timestamps = timestamps[:len(X_train)]
    test_timestamps = timestamps[len(X_train):]

    # 設置回調
    callbacks = [
        EarlyStopping(
            monitor='val_loss',  # 改為監控val_loss
            patience=25,         # 增加耐心值
            restore_best_weights=True,
            mode='min'
        ),
        ModelCheckpoint(
            filepath=model_checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        CSVLogger(training_log_path),
        EventEvaluationCallback(
            validation_data=(X_test, y_test),
            timestamps=test_timestamps  # 使用測試集的時間戳
        )
    ]

    # 調整batch size和epochs
    history = model.fit(
        X_train, y_train,
        epochs=15,          # 增加epochs
        batch_size=48,      # 調整batch size
        validation_split=0.2,
        class_weight={0: 1.0, 1: 2.5},  # 微調類別權重
        callbacks=callbacks,
        verbose=1
    )

    # 評估模型
    test_metrics, best_threshold, target_lead = evaluate_predictions(y_test, model.predict(X_test), test_timestamps, find_best_threshold=FIND_BEST_THRESHOLD)

    print(f"\n使用最佳閾值 {best_threshold:.2f} 的測試指標:")
    print(f"detections: {test_metrics['detections']} / {test_metrics['detections'] + test_metrics['missed_events']}")
    print(f"early_detections: {test_metrics['early_detections']}")
    print(f"late_detections: {test_metrics['late_detections']}")
    print(f"missed_events: {test_metrics['missed_events']}")
    print(f"false_alarms: {test_metrics['false_alarms']}")

    # 計算並顯示時間差異統計
    if test_metrics['early_time_diffs']:
        avg_early_time = np.mean(test_metrics['early_time_diffs'])
        print(f"平均提前時間: {avg_early_time:.2f} 秒 (目標: {target_lead:.1f} 秒)")
        print(f"提前檢測率: {(test_metrics['early_detections'] / max(test_metrics['detections'], 1)) * 100:.2f}%")
        
        # 計算時間準確率（使用秒數範圍）
        time_diff_seconds = abs(avg_early_time - target_lead)
        if time_diff_seconds <= 3:  # 在目標時間±3秒範圍內
            time_accuracy = 100.0
        else:
            # 超出範圍，根據秒數差距計算準確性（百分比）
            time_accuracy = max(0, 100.0 * (1.0 - (time_diff_seconds - 3) / 7))
        
        print(f"時間準確率: {time_accuracy:.2f}% (±3秒內為100%)")
    else:
        print("沒有提前檢測記錄")

    if test_metrics['late_time_diffs']:
        avg_late_time = np.mean(test_metrics['late_time_diffs'])
        print(f"平均延遲時間: {avg_late_time:.2f} 秒")
        print(f"延遲檢測率: {(test_metrics['late_detections'] / max(test_metrics['detections'], 1)) * 100:.2f}%")

    # 保存最終模型
    model.save(final_model_path)

    # 繪製時間軸上的事件對比圖
    plot_event_timeline(y_test, model.predict(X_test), test_timestamps, threshold=best_threshold)

    # 繪製詳細的事件時間差異對比圖
    plot_event_time_differences(y_test, model.predict(X_test), test_timestamps, threshold=best_threshold)

    # 使用不同窗口大小繪製準確度趨勢圖
    for window_size in [20, 50, 100]:
        plot_prediction_accuracy_trend(y_test, model.predict(X_test), test_timestamps, threshold=best_threshold, window_size=window_size)

    # 繪製預測時間差異分布圖
    plot_prediction_time_diff_distribution(y_test, model.predict(X_test), test_timestamps, threshold=best_threshold)

    # 生成評估報告
    evaluation_summary = generate_evaluation_summary(y_test, model.predict(X_test), test_timestamps, threshold=best_threshold)

    # 可視化訓練過程
    plt.figure(figsize=(12, 4))

    # 繪製準確率
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # 繪製損失
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(training_history_path)
    plt.close()

except Exception as e:
    print(f"數據處理錯誤: {e}")
    raise



