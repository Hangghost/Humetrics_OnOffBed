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
        
    def on_epoch_end(self, epoch, logs=None):
        x_val, y_val = self.validation_data
        y_pred = self.model.predict(x_val)
        metrics, threshold = evaluate_predictions(y_val, y_pred, self.timestamps)
        
        if metrics is not None:
            print(f"\nEpoch {epoch + 1} - Evaluation Metrics (threshold={threshold:.2f}):")
            print(f"Detections: {metrics['detections']}")
            print(f"Missed Events: {metrics['missed_events']}")
            print(f"False Alarms: {metrics['false_alarms']}")
        else:
            print(f"\nEpoch {epoch + 1} - No valid metrics available")

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
    sys.exit()
    
    return events

def create_event_labels(df, events):
    """為每個時間點創建標籤"""
    labels = np.zeros(len(df))
    
    for event in events:
        event_time = event['time']
        # 只設置單一預警時間的標籤
        start_idx = max(0, event_time - WARNING_TIME)
        end_idx = event_time
        labels[start_idx:end_idx] = 1
            
    return labels

def create_sequences(df, cleaned_data_path, use_sum_only=SUM_ONLY):
    """修改後的序列創建函數，增加use_sum_only參數"""
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
    # 創建標籤
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
    
    # 分別收集正負樣本
    positive_sequences = []
    positive_labels = []
    negative_sequences = []
    negative_labels = []
    
    for i in range(0, len(df) - WINDOW_SIZE + 1, STEP_SIZE):
        window = df_features.iloc[i:(i + WINDOW_SIZE)]
        label = event_labels[i + WINDOW_SIZE - 1]
        
        if label == 1:
            positive_sequences.append(window.values.astype('float64'))
            positive_labels.append(label)
        else:
            negative_sequences.append(window.values.astype('float64'))
            negative_labels.append(label)
    
    # 對負樣本進行下採樣
    neg_sample_size = len(positive_sequences) * 3  # 保持1:3的比例
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
    """修改後的評估函數"""
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
        
        # 簡化的評估指標
        metrics = {
            'detections': 0,    # 成功預測
            'missed_events': 0, # 漏報
            'false_alarms': 0   # 誤報
        }
        
        # 評估預測結果
        for actual_event in actual_events:
            event_detected = False
            
            for pred_event in events_detected:
                time_diff = actual_event['event_time'] - pred_event['prediction_time']
                
                if 0 < time_diff <= WARNING_TIME:
                    event_detected = True
                    metrics['detections'] += 1
                    break
            
            if not event_detected:
                metrics['missed_events'] += 1
        
        # 計算誤報
        for pred_event in events_detected:
            matched = False
            for actual_event in actual_events:
                time_diff = actual_event['event_time'] - pred_event['prediction_time']
                if 0 < time_diff <= WARNING_TIME:
                    matched = True
                    break
            if not matched:
                metrics['false_alarms'] += 1
        
        # 計算評估分數
        detection_rate = metrics['detections'] / (metrics['detections'] + metrics['missed_events'] + 1e-6)
        false_alarm_rate = metrics['false_alarms'] / (metrics['detections'] + 1e-6)
        f1_score = 2 * (detection_rate * (1 - false_alarm_rate)) / (detection_rate + (1 - false_alarm_rate) + 1e-6)
        
        return metrics, f1_score

    if find_best_threshold:
        thresholds = np.arange(0.3, 0.7, 0.02)
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
        
        if best_metrics is None:
            best_metrics, _ = evaluate_with_threshold(0.5)
        
        print(f"\n最佳閾值: {best_threshold:.2f}, 最佳分數: {best_score:.4f}")
        return best_metrics, best_threshold
    else:
        metrics, _ = evaluate_with_threshold(0.5)
        return metrics, 0.5

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
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00025),
        loss='binary_crossentropy',
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

# 編譯模型
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy', 'recall', 'precision']
)

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
    epochs=5,          # 增加epochs
    batch_size=48,      # 調整batch size
    validation_split=0.2,
    class_weight={0: 1.0, 1: 2.5},  # 微調類別權重
    callbacks=callbacks,
    verbose=1
)

# 評估模型
test_metrics = model.evaluate(X_test, y_test)
print("\nTest Metrics:")
print(f"Loss: {test_metrics[0]:.4f}")
print(f"Accuracy: {test_metrics[1]:.4f}")
print(f"Recall: {test_metrics[2]:.4f}")
print(f"Precision: {test_metrics[3]:.4f}")

# 在測試集上進行預測和評估
y_pred = model.predict(X_test)
test_metrics, best_threshold = evaluate_predictions(y_test, y_pred, test_timestamps)

print(f"\n使用最佳閾值 {best_threshold:.2f} 的測試指標:")
print(f"Detections: {test_metrics['detections']}")
print(f"Missed Events: {test_metrics['missed_events']}")
print(f"False Alarms: {test_metrics['false_alarms']}")

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



