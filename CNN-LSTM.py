import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, LSTM, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from tensorflow.keras.layers import Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
import os
import tensorflow as tf

# 設定隨機種子
np.random.seed(1337)

# 設定參數
WINDOW_SIZE = 15  # 15秒的窗口
OVERLAP = 0.8    # 80% 重疊
STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP))  # 滑動步長

# 預警時間設定
WARNING_TIMES = {
    'EARLY': 15,    # 15秒預警
    'IMMEDIATE': 5, # 5秒預警
    'CRITICAL': 2   # 2秒預警
}

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
        metrics = evaluate_predictions(y_val, y_pred, self.timestamps)
        
        print(f"\nEpoch {epoch + 1} - Evaluation Metrics:")
        print(f"Early Detections: {metrics['early_detections']}")
        print(f"Immediate Detections: {metrics['immediate_detections']}")
        print(f"Critical Detections: {metrics['critical_detections']}")
        print(f"Missed Events: {metrics['missed_events']}")
        print(f"False Alarms: {metrics['false_alarms']}")

def get_cleaned_data_path(raw_data_path):
    """根據原始數據路徑生成清理後數據的路徑"""
    # 獲取原始檔案名（不含路徑）
    raw_filename = os.path.basename(raw_data_path)
    # 在檔名前加上 'cleaned_' 前綴
    cleaned_filename = f"cleaned_{raw_filename}"
    # 組合完整路徑
    return os.path.join("./_data/training", cleaned_filename)

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
    """檢測離床和上床事件"""
    events = []
    status_changes = df['OnBed_Status'].diff()
    
    # 找出所有狀態變化的時間點
    for idx in range(1, len(df)):
        if status_changes.iloc[idx] == -1:  # 1->0 離床
            events.append({
                'time': idx,
                'type': 'leaving',
                'original_status': 1
            })
        elif status_changes.iloc[idx] == 1:  # 0->1 上床
            events.append({
                'time': idx,
                'type': 'entering',
                'original_status': 0
            })
    
    return events

def create_event_labels(df, events):
    """為每個時間點創建標籤"""
    labels = np.zeros(len(df))
    
    for event in events:
        event_time = event['time']
        
        # 為每個預警時間點設置標籤
        for warning_time in WARNING_TIMES.values():
            start_idx = max(0, event_time - warning_time)
            end_idx = event_time
            labels[start_idx:end_idx] = 1
            
    return labels

def create_sequences(df, cleaned_data_path):
    """修改後的序列創建函數"""
    sequences = []
    labels = []
    
    # 檢測事件
    events = detect_bed_events(df)
    # 創建標籤
    event_labels = create_event_labels(df, events)
    
    # 只保留需要的列
    required_columns = [f'Channel_{i}_Raw' for i in range(1, 7)]
    
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"缺少必要的列: {[col for col in required_columns if col not in df.columns]}")
    
    df_features = df[required_columns].copy()
    
    # 存一份CSV
    df_features.to_csv(cleaned_data_path, index=False)
    
    # 創建序列
    for i in range(0, len(df) - WINDOW_SIZE + 1, STEP_SIZE):
        window = df_features.iloc[i:(i + WINDOW_SIZE)]
        label = event_labels[i + WINDOW_SIZE - 1]  # 使用窗口最後一個時間點的標籤
        
        try:
            raw_values = window.values.astype('float64')
            sequences.append(raw_values)
            labels.append(label)
            
        except Exception as e:
            print(f"處理窗口 {i} 時發生錯誤: {e}")
            continue
    
    sequences = np.array(sequences)
    labels = np.array(labels)
    
    # 保存處理後的數據
    save_processed_sequences(sequences, labels, cleaned_data_path)
    
    return sequences, labels

def load_and_process_data(raw_data_path):
    """載入並處理數據，如果有清理過的數據則直接讀取"""
    # 獲取對應的清理後數據路徑
    cleaned_data_path = get_cleaned_data_path(raw_data_path)
    sequences_path = cleaned_data_path.replace('.csv', '.npz')
    
    # 檢查是否存在已處理的序列資料
    if os.path.exists(sequences_path):
        try:
            # 直接載入處理好的序列資料
            data = np.load(sequences_path)
            sequences = data['sequences']
            labels = data['labels']
            print(f"發現已處理的序列資料，直接讀取: {sequences_path}")
            return sequences, labels
        except Exception as e:
            print(f"讀取序列資料時發生錯誤: {e}")
            print("將重新處理原始數據...")
    
    # 如果沒有處理好的序列資料，則從頭處理
    try:
        dataset = pd.read_csv(raw_data_path)
        print(f"數據集形狀: {dataset.shape}")
        print(f"數據集列: {dataset.columns.tolist()}")
        return create_sequences(dataset, cleaned_data_path)
    except Exception as e:
        print(f"數據處理錯誤: {e}")
        raise

def evaluate_predictions(y_true, y_pred, timestamps, threshold=0.5):
    """
    評估預測結果
    
    參數:
    - y_true: 真實標籤
    - y_pred: 預測機率
    - timestamps: 對應的時間戳
    - threshold: 預測閾值
    
    返回:
    - 評估指標字典
    """
    predictions = (y_pred >= threshold).astype(int)
    events_detected = []
    
    # 找出所有預測事件
    i = 0
    while i < len(predictions):
        if predictions[i] == 1:
            # 找出連續預測的起始和結束
            start_idx = i
            while i < len(predictions) and predictions[i] == 1:
                i += 1
            end_idx = i - 1
            
            # 記錄預測事件
            events_detected.append({
                'start_time': timestamps[start_idx],
                'end_time': timestamps[end_idx],
                'prediction_time': timestamps[start_idx]  # 使用最早的預測時間
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
                'event_time': timestamps[end_idx]  # 實際事件發生時間
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
    
    # 配對預測事件和實際事件
    for actual_event in actual_events:
        event_detected = False
        best_prediction_time = None
        
        for pred_event in events_detected:
            time_diff = actual_event['event_time'] - pred_event['prediction_time']
            
            # 只考慮提前預測的情況
            if 0 < time_diff <= WARNING_TIMES['EARLY']:
                event_detected = True
                if best_prediction_time is None or pred_event['prediction_time'] < best_prediction_time:
                    best_prediction_time = pred_event['prediction_time']
        
        if event_detected and best_prediction_time is not None:
            time_diff = actual_event['event_time'] - best_prediction_time
            
            # 根據預測提前時間分類
            if time_diff >= WARNING_TIMES['EARLY']:
                metrics['early_detections'] += 1
            elif time_diff >= WARNING_TIMES['IMMEDIATE']:
                metrics['immediate_detections'] += 1
            elif time_diff >= WARNING_TIMES['CRITICAL']:
                metrics['critical_detections'] += 1
        else:
            metrics['missed_events'] += 1
    
    # 計算誤報數（未匹配到實際事件的預測）
    for pred_event in events_detected:
        matched = False
        for actual_event in actual_events:
            time_diff = actual_event['event_time'] - pred_event['prediction_time']
            if 0 < time_diff <= WARNING_TIMES['EARLY']:
                matched = True
                break
        if not matched:
            metrics['false_alarms'] += 1
    
    return metrics


# 修改原本的數據載入部分
try:
    X, y = load_and_process_data("./_data/SPS2021PA000015_20241227_04_20241228_04_data.csv")
    print(f"特徵形狀: {X.shape}")
    print(f"標籤形狀: {y.shape}")
except Exception as e:
    print(f"數據處理錯誤: {e}")
    raise

# 分割訓練和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 創建日誌目錄
log_dir = '_logs/bed_monitor'
os.makedirs(log_dir, exist_ok=True)

# 定義檔案路徑
model_checkpoint_path = os.path.join(log_dir, 'model-{epoch:02d}-{val_accuracy:.4f}.keras')
training_log_path = os.path.join(log_dir, 'training.csv')
final_model_path = os.path.join(log_dir, 'final_model.keras')
training_history_path = os.path.join(log_dir, 'training_history.png')

# 建立模型
input_layer = Input(shape=(WINDOW_SIZE, X_train.shape[2]))
x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(input_layer)
x = BatchNormalization()(x)
x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)
x = Dropout(0.2)(x)

x = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)
x = Dropout(0.3)(x)

x = LSTM(units=256, return_sequences=True)(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = LSTM(units=128)(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

output_layer = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input_layer, outputs=output_layer)

# 編譯模型
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 生成完整的時間戳序列
timestamps = np.arange(len(X))  # 使用完整數據集的長度

# 分割訓練集和測試集的時間戳
train_timestamps = timestamps[:len(X_train)]
test_timestamps = timestamps[len(X_train):]

# 設置回調
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True
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

# 訓練模型
history = model.fit(
    X_train, y_train,
    epochs=10,  # 增加訓練輪數
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# 評估模型
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\nTest accuracy: {test_accuracy:.4f}")

# 在測試集上進行預測和評估
y_pred = model.predict(X_test)
test_metrics = evaluate_predictions(y_test, y_pred, test_timestamps)  # 使用測試集的時間戳

print("\nFinal Test Metrics:")
print(f"Early Detections (15s): {test_metrics['early_detections']}")
print(f"Immediate Detections (5s): {test_metrics['immediate_detections']}")
print(f"Critical Detections (2s): {test_metrics['critical_detections']}")
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



