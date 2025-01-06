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

# 設定隨機種子
np.random.seed(1337)

# 設定參數
WINDOW_SIZE = 15  # 15秒的窗口
OVERLAP = 0.8    # 80% 重疊
STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP))  # 滑動步長

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
    # 將 .csv 副檔名改為 .npz
    sequences_path = cleaned_data_path.replace('.csv', '.npz')
    np.savez(sequences_path, sequences=sequences, labels=labels)
    print(f"序列資料已保存至: {sequences_path}")

def create_sequences(df, cleaned_data_path):
    """創建時間序列窗口"""
    sequences = []
    labels = []
    
    # 先檢查數據
    print(f"原始數據長度: {len(df)}")
    
    # 只保留需要的列
    required_columns = [f'Channel_{i}_Raw' for i in range(1, 7)] + ['OnBed_Status']
    
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"缺少必要的列: {[col for col in required_columns if col not in df.columns]}")
    
    # 使用 .copy() 創建數據的深度複製
    df = df[required_columns].copy()
    
    # 確保數據類型為 float64
    for col in required_columns[:-1]:  # 除了 OnBed_Status
        df.loc[:, col] = df[col].astype('float64')
    
    # 檢查數值是否有效
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"清理後數據長度: {len(df)}")
    
    # 計算特徵
    for i in range(0, len(df) - WINDOW_SIZE + 1, STEP_SIZE):
        window = df.iloc[i:(i + WINDOW_SIZE)]
        
        try:
            # 只使用原始壓力值
            raw_values = window[[f'Channel_{i}_Raw' for i in range(1, 7)]].values.astype('float64')
            
            sequences.append(raw_values)
            labels.append(window['OnBed_Status'].iloc[-1])
            
        except Exception as e:
            print(f"處理窗口 {i} 時發生錯誤: {e}")
            continue
    
    print(f"生成序列數量: {len(sequences)}")
    
    if len(sequences) == 0:
        raise ValueError("沒有生成任何有效的序列")
    
    sequences = np.array(sequences)
    labels = np.array(labels)
    
    # 保存清理後的原始數據和序列資料
    os.makedirs(os.path.dirname(cleaned_data_path), exist_ok=True)
    df.to_csv(cleaned_data_path, index=False)
    save_processed_sequences(sequences, labels, cleaned_data_path)
    
    return sequences, labels

def load_and_process_data(raw_data_path):
    """載入並處理數據，如果有清理過的數據則直接讀取"""
    # 獲取對應的清理後數據路徑
    cleaned_data_path = get_cleaned_data_path(raw_data_path)

    # Save a CSV file for test
    df = pd.read_csv(raw_data_path)
    df.to_csv(cleaned_data_path, index=False)
    
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
    CSVLogger(training_log_path)
]

# 訓練模型
history = model.fit(
    X_train, y_train,
    epochs=1,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# 評估模型
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\nTest accuracy: {test_accuracy:.4f}")

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