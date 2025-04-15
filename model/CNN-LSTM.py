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
# 設定隨機種子
np.random.seed(1337)

# 設定參數
WINDOW_SIZE = 15  # 15秒的窗口
OVERLAP = 0.8    # 80% 重疊
STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP))  # 滑動步長

# 修改預警時間設定
WARNING_TIME = 15  # 設定單一預警時間（秒）

INPUT_DATA_PATH = "./_data/pyqt_viewer/SPS2025PA000146_20250406_04_20250407_04_data.csv"
TRAINING_LOG_PATH = "training_test_sum.csv"
FINAL_MODEL_PATH = "final_model_test_sum.keras"
TRAINING_HISTORY_PATH = "training_history_test_sum.png"
LOG_DIR = "./_logs/bed_monitor_test_sum"

FIND_BEST_THRESHOLD = False
SUM_ONLY = False
SILENCE_TIME = 180 


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

def save_processed_sequences(sequences, labels, cleaned_data_path):
    """保存處理後的序列資料"""
    # # 先存一份CSV
    # df = pd.DataFrame(sequences)
    # df.to_csv(cleaned_data_path, index=False)

    # 將 .csv 副檔名改為 .npz
    sequences_path = cleaned_data_path.replace('.csv', '.npz')
    np.savez(sequences_path, sequences=sequences, labels=labels)
    print(f"序列資料已保存至: {sequences_path}")

def detect_bed_events(df):
    """檢測離床事件，並在事件發生後的靜默時間內不檢測新事件"""
    events = []
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
            last_event_time = idx  # 更新最後事件時間

    print(f"檢測到 {len(events)} 個離床事件")
    
    return events

def create_event_labels(df, events):
    """為每個時間點創建漸進式標籤，使模型學會提前預測"""
    labels = np.zeros(len(df))
    
    for event in events:
        event_time = event['time']
        
        # 預警時間範圍
        start_idx = max(0, event_time - WARNING_TIME)
        end_idx = event_time
        
        # 創建漸進式標籤：越接近實際事件，標籤值越高
        # 這將鼓勵模型在接近事件時預測概率更高
        for i in range(start_idx, end_idx):
            # 計算到事件的時間距離
            time_to_event = end_idx - i
            
            # 標籤值從0.5漸進到1.0
            # 離事件越近，標籤值越接近1.0
            # 這使得模型學習在接近事件時提高預測概率
            normalized_distance = 1.0 - (time_to_event / WARNING_TIME)
            labels[i] = 0.5 + (0.5 * normalized_distance)
        
        # 事件發生時刻設為1.0
        labels[end_idx] = 1.0
            
    return labels

def create_sequences(df, cleaned_data_path, use_sum_only=SUM_ONLY):
    """修改後的序列創建函數，使用漸進式標籤"""
    sequences = []
    labels = []
    
    # 增強特徵工程
    df['pressure_sum'] = df[[f'Channel_{i}_Raw' for i in range(1, 7)]].sum(axis=1)
    df['pressure_std'] = df[[f'Channel_{i}_Raw' for i in range(1, 7)]].std(axis=1)
    df['pressure_change'] = df['pressure_sum'].diff()
    df['pressure_rolling_mean'] = df['pressure_sum'].rolling(window=5).mean()
    df['pressure_rolling_std'] = df['pressure_sum'].rolling(window=5).std()
    df['pressure_acceleration'] = df['pressure_change'].diff()
    
    # 填充NaN值
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # 檢測事件
    events = detect_bed_events(df)
    # 創建漸進式標籤
    event_labels = create_event_labels(df, events)
    
    # 根據use_sum_only參數選擇使用的特徵
    if use_sum_only:
        df_features = pd.DataFrame({'pressure_sum': df['pressure_sum']})
    else:
        required_columns = [f'Channel_{i}_Raw' for i in range(1, 7)]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"缺少必要的列: {[col for col in required_columns if col not in df.columns]}")
        df_features = df[required_columns].copy()
    
    # 存一份CSV
    df_features.to_csv(cleaned_data_path, index=False)
    
    # 分別收集正負樣本（正樣本包括所有標籤 >= 0.5 的樣本）
    positive_sequences = []
    positive_labels = []
    negative_sequences = []
    negative_labels = []
    
    for i in range(0, len(df) - WINDOW_SIZE + 1, STEP_SIZE):
        window = df_features.iloc[i:(i + WINDOW_SIZE)]
        label = event_labels[i + WINDOW_SIZE - 1]
        
        if label >= 0.5:  # 所有標籤 >= 0.5 都視為正樣本
            positive_sequences.append(window.values.astype('float64'))
            positive_labels.append(label)
        else:
            negative_sequences.append(window.values.astype('float64'))
            negative_labels.append(label)
    
    # 對負樣本進行下採樣
    neg_sample_size = len(positive_sequences) * 2  # 降低負樣本比例，更重視正樣本
    if len(negative_sequences) > neg_sample_size:
        indices = np.random.choice(len(negative_sequences), neg_sample_size, replace=False)
        negative_sequences = [negative_sequences[i] for i in indices]
        negative_labels = [negative_labels[i] for i in indices]
    
    # 合併正負樣本
    sequences = positive_sequences + negative_sequences
    labels = positive_labels + negative_labels
    
    # 打亂順序
    combined = list(zip(sequences, labels))
    np.random.shuffle(combined)
    sequences, labels = zip(*combined)
    
    # 保存處理後的數據
    save_processed_sequences(sequences, labels, cleaned_data_path)
    
    return sequences, labels

def load_and_process_data(raw_data_path, use_sum_only=SUM_ONLY):
    """載入並處理數據，增加use_sum_only參數"""
    # 獲取對應的清理後數據路徑
    cleaned_data_path = get_cleaned_data_path(raw_data_path)
    # 根據use_sum_only參數修改檔案名
    if use_sum_only:
        sequences_path = cleaned_data_path.replace('.csv', '_sum_only.npz')
    else:
        sequences_path = cleaned_data_path.replace('.csv', '.npz')
    
    # 檢查是否存在已處理的序列資料
    if os.path.exists(sequences_path):
        try:
            data = np.load(sequences_path)
            sequences = data['sequences']
            labels = data['labels']
            print(f"發現已處理的序列資料，直接讀取: {sequences_path}")
            return sequences, labels
        except Exception as e:
            print(f"讀取序列資料時發生錯誤: {e}")
            print("將重新處理原始數據...")
    
    try:
        dataset = pd.read_csv(raw_data_path)
        print(f"數據集形狀: {dataset.shape}")
        print(f"數據集列: {dataset.columns.tolist()}")
        return create_sequences(dataset, cleaned_data_path, use_sum_only=SUM_ONLY)
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
    
    # 確保所有要相加的層具有相同的通道數
    conv1 = Conv1D(64, kernel_size=5, padding='same', activation='relu')(input_layer)  # 改為64
    conv1 = BatchNormalization()(conv1)
    conv2 = Conv1D(64, kernel_size=3, padding='same', activation='relu')(conv1)  # 保持64
    conv2 = BatchNormalization()(conv2)
    conv2 = Add()([conv1, conv2])  # 現在兩者都是64通道
    
    # 第二個殘差塊
    conv3 = Conv1D(64, kernel_size=3, padding='same', activation='relu')(conv2)  # 保持64
    conv3 = BatchNormalization()(conv3)
    conv3 = Add()([conv2, conv3])  # 現在都是64通道
    
    x = MaxPooling1D(pool_size=2)(conv3)
    x = Dropout(0.25)(x)
    
    # LSTM層保持不變
    x = Bidirectional(LSTM(320, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    
    x = Bidirectional(LSTM(160))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    
    # 確保密集層的維度匹配
    dense1 = Dense(64, activation='relu')(x)  # 改為64
    dense1 = BatchNormalization()(dense1)
    dense2 = Dense(64, activation='relu')(dense1)  # 保持64
    dense2 = BatchNormalization()(dense2)
    dense2 = Add()([dense1, dense2])  # 現在都是64維
    
    output_layer = Dense(1, activation='sigmoid')(dense2)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    # 使用純時間差異損失函數
    def loss_wrapper(y_true, y_pred):
        return pure_time_difference_loss(y_true, y_pred, target_lead_time=WARNING_TIME/2)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00025),
        loss=loss_wrapper,  # 使用時間差異損失
        metrics=['accuracy', 'recall', 'precision']
    )
    
    return model

# 修改原本的數據載入部分
try:
    # 使用總和值進行訓練 
    sequences, labels = load_and_process_data(
        INPUT_DATA_PATH,
        use_sum_only=SUM_ONLY
    )
    X = np.array(sequences)
    y = np.array(labels)
    print(f"特徵形狀: {X.shape}")
    print(f"標籤形狀: {y.shape}")
except Exception as e:
    print(f"數據處理錯誤: {e}")
    raise

# 分割訓練和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=80)

# 創建日誌目錄
os.makedirs(LOG_DIR, exist_ok=True)

# 定義檔案路徑
model_checkpoint_path = os.path.join(LOG_DIR, 'model-{epoch:02d}-{val_accuracy:.4f}.keras')
training_log_path = os.path.join(LOG_DIR, TRAINING_LOG_PATH)
final_model_path = os.path.join(LOG_DIR, FINAL_MODEL_PATH)
training_history_path = os.path.join(LOG_DIR, TRAINING_HISTORY_PATH)

# 建立模型
model = build_model((WINDOW_SIZE, X_train.shape[2]))

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
    epochs=50,          # 增加epochs
    batch_size=48,      # 調整batch size
    validation_split=0.2,
    class_weight={0: 1.0, 1: 2.5},  # 微調類別權重
    callbacks=callbacks,
    verbose=1
)

# 評估模型
test_metrics, best_threshold, target_lead = evaluate_predictions(y_test, model.predict(X_test), test_timestamps, find_best_threshold=FIND_BEST_THRESHOLD)

print(f"\n使用最佳閾值 {best_threshold:.2f} 的測試指標:")
print(f"總檢測成功: {test_metrics['detections']} / {test_metrics['detections'] + test_metrics['missed_events']}")
print(f"提前檢測: {test_metrics['early_detections']}")
print(f"延遲檢測: {test_metrics['late_detections']}")
print(f"漏報: {test_metrics['missed_events']}")
print(f"誤報: {test_metrics['false_alarms']}")

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



